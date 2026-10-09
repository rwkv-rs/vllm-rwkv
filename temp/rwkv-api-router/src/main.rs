// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Streaming reverse proxy; response bodies remain owned by Hyper, including
//! backpressure and cancellation when the downstream connection closes.

use std::borrow::Cow;
use std::error::Error as _;
use std::net::IpAddr;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context as _, bail};
use axum::body::{Body, to_bytes};
use axum::extract::{Path, Query, Request, State};
use axum::http::header::{AUTHORIZATION, CONNECTION, HOST, WWW_AUTHENTICATE};
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode, Uri};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{MethodFilter, get, on};
use axum::serve::ListenerExt as _;
use axum::{Json, Router};
use bytes::Bytes;
use clap::Parser;
use futures::future::try_join_all;
use http_body_util::{Full, LengthLimitError};
use hyper_util::client::legacy::{Client, connect::HttpConnector};
use hyper_util::rt::{TokioExecutor, TokioTimer};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;
use tokio::net::{TcpListener, TcpStream};

const MODEL_UPSTREAMS: &[(&str, &str)] = &[
    ("rwkv7-g1k-1.5b", "http://192.168.0.129:18001"),
    ("rwkv7-g1k-2.9b", "http://192.168.0.129:18002"),
    ("rwkv7-g1k-7.2b", "http://192.168.0.129:18003"),
    ("rwkv7-g1k-13.3b", "http://127.0.0.1:18004"),
];
const MAX_REQUEST_BYTES: usize = 16 * 1024 * 1024;

#[derive(Parser)]
struct Config {
    #[arg(long)]
    api_key_sha256_file: PathBuf,
    #[arg(long, default_value = "127.0.0.1")]
    host: IpAddr,
    #[arg(long, default_value_t = 18000)]
    port: u16,
    #[arg(long, default_value = "4")]
    worker_threads: NonZeroUsize,
    /// Override a model's HTTP upstream; repeat for multiple models.
    #[arg(long, value_name = "MODEL=URL")]
    upstream: Vec<String>,
}

struct AppState {
    client: Client<HttpConnector, Full<Bytes>>,
    api_key_sha256: [u8; 32],
    upstreams: Vec<(&'static str, Uri)>,
}

impl AppState {
    fn new(api_key_sha256: [u8; 32], upstreams: Vec<(&'static str, Uri)>) -> Self {
        let mut connector = HttpConnector::new();
        connector.set_connect_timeout(Some(Duration::from_secs(5)));
        connector.set_nodelay(true);
        let client = Client::builder(TokioExecutor::new())
            .pool_timer(TokioTimer::new())
            .pool_idle_timeout(Duration::from_secs(60))
            .pool_max_idle_per_host(4096)
            .build(connector);
        Self {
            client,
            api_key_sha256,
            upstreams,
        }
    }

    fn upstream(&self, model: &str) -> Result<&Uri, ApiError> {
        self.upstreams
            .iter()
            .find(|(name, _)| *name == model)
            .map(|(_, uri)| uri)
            .ok_or_else(|| error(StatusCode::NOT_FOUND, format!("Unknown model: {model}")))
    }
}

struct ApiError {
    status: StatusCode,
    message: String,
}

fn error(status: StatusCode, message: impl Into<String>) -> ApiError {
    ApiError {
        status,
        message: message.into(),
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let error_type = if self.status == StatusCode::UNAUTHORIZED {
            "authentication_error"
        } else if self.status.is_server_error() {
            "server_error"
        } else {
            "invalid_request_error"
        };
        let mut response = (
            self.status,
            Json(json!({"error": {"message": self.message, "type": error_type}})),
        )
            .into_response();
        if self.status == StatusCode::UNAUTHORIZED {
            response
                .headers_mut()
                .insert(WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
        }
        response
    }
}

fn verify_token(authorization: Option<&HeaderValue>, api_key_sha256: &[u8; 32]) -> bool {
    let Some(authorization) = authorization.and_then(|value| value.to_str().ok()) else {
        return false;
    };
    let Some(token) = authorization
        .strip_prefix("Bearer ")
        .filter(|token| !token.is_empty())
    else {
        return false;
    };
    let digest: [u8; 32] = Sha256::digest(token.as_bytes()).into();
    bool::from(digest.ct_eq(api_key_sha256))
}

async fn authenticate_api_key(
    State(state): State<Arc<AppState>>,
    request: Request,
    next: Next,
) -> Result<Response, ApiError> {
    if !verify_token(request.headers().get(AUTHORIZATION), &state.api_key_sha256) {
        return Err(error(StatusCode::UNAUTHORIZED, "Unauthorized"));
    }
    Ok(next.run(request).await)
}

fn build_router(state: Arc<AppState>) -> Router {
    let protected = Router::new()
        .route("/metrics", get(metrics))
        .route("/v1/models", get(models))
        .route("/v1/models/{model}", get(model))
        .route(
            "/v1/{*path}",
            on(
                MethodFilter::GET
                    .or(MethodFilter::POST)
                    .or(MethodFilter::PUT)
                    .or(MethodFilter::PATCH)
                    .or(MethodFilter::DELETE)
                    .or(MethodFilter::OPTIONS),
                proxy,
            ),
        )
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            authenticate_api_key,
        ));
    Router::new()
        .route("/health", get(|| async { Json(json!({"status": "ok"})) }))
        .merge(protected)
        .with_state(state)
}

fn strip_hop_by_hop(headers: &mut HeaderMap) {
    let connection_headers: Vec<HeaderName> = headers
        .get_all(CONNECTION)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .filter_map(|name| HeaderName::from_bytes(name.trim().as_bytes()).ok())
        .collect();
    for name in connection_headers {
        headers.remove(name);
    }
    for name in [
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    ] {
        headers.remove(name);
    }
}

async fn forward(
    state: &AppState,
    upstream: &Uri,
    mut request: hyper::Request<Full<Bytes>>,
) -> Result<Response, ApiError> {
    *request.uri_mut() = Uri::builder()
        .scheme("http")
        .authority(upstream.authority().expect("upstream authority").clone())
        .path_and_query(
            request
                .uri()
                .path_and_query()
                .expect("request path")
                .clone(),
        )
        .build()
        .expect("validated upstream and request URI");
    strip_hop_by_hop(request.headers_mut());
    request.headers_mut().remove(AUTHORIZATION);
    request.headers_mut().remove(HOST);
    let mut response = state.client.request(request).await.map_err(|err| {
        eprintln!("upstream request failed: {err:?}");
        error(StatusCode::BAD_GATEWAY, "Upstream request failed")
    })?;
    strip_hop_by_hop(response.headers_mut());
    Ok(response.map(Body::new))
}

fn get_request(path: &str) -> hyper::Request<Full<Bytes>> {
    hyper::Request::builder()
        .uri(path)
        .body(Full::default())
        .expect("internal GET request")
}

#[derive(Deserialize)]
struct MetricsQuery {
    model: Option<String>,
}

async fn metrics(
    State(state): State<Arc<AppState>>,
    Query(query): Query<MetricsQuery>,
) -> Result<Response, ApiError> {
    let model = query
        .model
        .ok_or_else(|| error(StatusCode::BAD_REQUEST, "Query parameter model is required"))?;
    forward(&state, state.upstream(&model)?, get_request("/metrics")).await
}

async fn model(
    State(state): State<Arc<AppState>>,
    Path(model): Path<String>,
) -> Result<Response, ApiError> {
    forward(
        &state,
        state.upstream(&model)?,
        get_request(&format!("/v1/models/{model}")),
    )
    .await
}

#[derive(Deserialize)]
struct ModelList {
    data: Vec<Value>,
}

async fn models(State(state): State<Arc<AppState>>) -> Result<Response, ApiError> {
    let lists = try_join_all(state.upstreams.iter().map(|(_, upstream)| async {
        let response = forward(&state, upstream, get_request("/v1/models")).await?;
        if !response.status().is_success() {
            return Err(error(
                StatusCode::BAD_GATEWAY,
                "Upstream model listing failed",
            ));
        }
        let body = to_bytes(response.into_body(), MAX_REQUEST_BYTES)
            .await
            .map_err(|_| error(StatusCode::BAD_GATEWAY, "Failed to read upstream models"))?;
        serde_json::from_slice::<ModelList>(&body)
            .map_err(|_| error(StatusCode::BAD_GATEWAY, "Invalid upstream model listing"))
    }))
    .await?;
    let data: Vec<Value> = lists.into_iter().flat_map(|list| list.data).collect();
    Ok(Json(json!({"object": "list", "data": data})).into_response())
}

#[derive(Deserialize)]
struct Model<'a> {
    #[serde(borrow)]
    model: Option<Cow<'a, str>>,
}

async fn proxy(State(state): State<Arc<AppState>>, request: Request) -> Result<Response, ApiError> {
    let (parts, body) = request.into_parts();
    let body = to_bytes(body, MAX_REQUEST_BYTES).await.map_err(|err| {
        if err
            .source()
            .is_some_and(|source| source.is::<LengthLimitError>())
        {
            error(StatusCode::PAYLOAD_TOO_LARGE, "Request body is too large")
        } else {
            error(StatusCode::BAD_REQUEST, "Failed to read request body")
        }
    })?;
    if body.iter().find(|byte| !byte.is_ascii_whitespace()) != Some(&b'{') {
        return Err(error(
            StatusCode::BAD_REQUEST,
            "Request body must be a JSON object",
        ));
    }
    let model: Model<'_> = serde_json::from_slice(&body).map_err(|err| {
        let message = if err.is_data() {
            "Request body must contain a string model"
        } else {
            "Request body must be a JSON object"
        };
        error(StatusCode::BAD_REQUEST, message)
    })?;
    let model = model.model.ok_or_else(|| {
        error(
            StatusCode::BAD_REQUEST,
            "Request body must contain a string model",
        )
    })?;
    let upstream = state.upstream(&model)?;
    forward(
        &state,
        upstream,
        hyper::Request::from_parts(parts, Full::new(body)),
    )
    .await
}

fn configured_upstreams(overrides: &[String]) -> anyhow::Result<Vec<(&'static str, Uri)>> {
    let mut upstreams: Vec<_> = MODEL_UPSTREAMS
        .iter()
        .map(|&(model, url)| (model, url.parse::<Uri>().expect("static upstream URI")))
        .collect();
    for value in overrides {
        let (model, url) = value
            .split_once('=')
            .context("upstream must be MODEL=URL")?;
        let entry = upstreams
            .iter_mut()
            .find(|(name, _)| *name == model)
            .with_context(|| format!("Unknown model: {model}"))?;
        let uri: Uri = url.parse().context("invalid upstream URI")?;
        if uri.scheme_str() != Some("http")
            || uri.authority().is_none()
            || uri
                .authority()
                .is_some_and(|authority| authority.as_str().contains('@'))
            || uri.path() != "/"
            || uri.query().is_some()
        {
            bail!("upstream must be an HTTP origin without credentials, path, or query");
        }
        entry.1 = uri;
    }
    Ok(upstreams)
}

fn enable_tcp_nodelay(stream: &mut TcpStream) {
    if let Err(err) = stream.set_nodelay(true) {
        eprintln!("failed to enable TCP_NODELAY: {err}");
    }
}

async fn shutdown_signal() {
    let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .expect("install SIGTERM handler");
    tokio::select! {
        result = tokio::signal::ctrl_c() => result.expect("receive SIGINT"),
        _ = terminate.recv() => {},
    }
}

fn main() -> anyhow::Result<()> {
    let config = Config::parse();
    let api_key_sha256 = std::fs::read(&config.api_key_sha256_file)
        .context("read API key SHA-256 verifier")?
        .try_into()
        .map_err(|_| anyhow::anyhow!("API key SHA-256 verifier must be exactly 32 bytes"))?;
    let upstreams = configured_upstreams(&config.upstream)?;
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(config.worker_threads.get())
        .enable_all()
        .build()?
        .block_on(async {
            let state = Arc::new(AppState::new(api_key_sha256, upstreams));
            let listener = TcpListener::bind((config.host, config.port)).await?;
            eprintln!("RWKV API router listening on {}", listener.local_addr()?);
            axum::serve(listener.tap_io(enable_tcp_nodelay), build_router(state))
                .with_graceful_shutdown(shutdown_signal())
                .await
        })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;
    use std::net::SocketAddr;
    use std::pin::Pin;
    use std::task::{Context, Poll};

    use axum::extract::ConnectInfo;
    use axum::http::Method;
    use axum::routing::post;
    use futures::stream;
    use http_body_util::BodyExt as _;
    use hyper::body::{Body as HttpBody, Frame};
    use tokio::sync::{Barrier, Mutex, mpsc, oneshot};
    use tokio::task::JoinHandle;
    use tokio::time::timeout;
    use tokio_stream::wrappers::ReceiverStream;
    use tower::ServiceExt as _;

    use super::*;

    const MODEL: &str = "rwkv7-g1k-7.2b";
    const TOKEN: &str = "test-secret";
    const TEST_TIMEOUT: Duration = Duration::from_secs(5);

    struct TestServer {
        uri: Uri,
        task: JoinHandle<()>,
    }

    impl TestServer {
        async fn start(app: Router) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let uri = format!("http://{}", listener.local_addr().unwrap())
                .parse()
                .unwrap();
            let task = tokio::spawn(async move {
                axum::serve(
                    listener.tap_io(enable_tcp_nodelay),
                    app.into_make_service_with_connect_info::<SocketAddr>(),
                )
                .await
                .unwrap();
            });
            Self { uri, task }
        }

        fn app(&self) -> Router {
            build_router(Arc::new(AppState::new(
                Sha256::digest(TOKEN.as_bytes()).into(),
                vec![(MODEL, self.uri.clone())],
            )))
        }
    }

    impl Drop for TestServer {
        fn drop(&mut self) {
            self.task.abort();
        }
    }

    fn request(method: Method, path: &str, body: impl Into<Body>) -> Request {
        Request::builder()
            .method(method)
            .uri(path)
            .header(AUTHORIZATION, format!("Bearer {TOKEN}"))
            .body(body.into())
            .unwrap()
    }

    fn completion_request() -> Request {
        request(
            Method::POST,
            "/v1/chat/completions",
            format!(r#"{{"model":"{MODEL}"}}"#),
        )
    }

    async fn response_json(response: Response) -> Value {
        let body = to_bytes(response.into_body(), MAX_REQUEST_BYTES)
            .await
            .unwrap();
        serde_json::from_slice(&body).unwrap()
    }

    // A response which stays open until the proxy cancels it.
    struct PendingBody {
        first: Option<Bytes>,
        dropped: Option<oneshot::Sender<()>>,
    }

    impl PendingBody {
        fn new(dropped: Option<oneshot::Sender<()>>) -> Self {
            Self {
                first: Some(Bytes::from_static(b"data: first\n\n")),
                dropped,
            }
        }
    }

    impl HttpBody for PendingBody {
        type Data = Bytes;
        type Error = Infallible;

        fn poll_frame(
            mut self: Pin<&mut Self>,
            _: &mut Context<'_>,
        ) -> Poll<Option<Result<Frame<Bytes>, Infallible>>> {
            match self.first.take() {
                Some(bytes) => Poll::Ready(Some(Ok(Frame::data(bytes)))),
                None => Poll::Pending,
            }
        }
    }

    impl Drop for PendingBody {
        fn drop(&mut self) {
            if let Some(dropped) = self.dropped.take() {
                let _ = dropped.send(());
            }
        }
    }

    #[tokio::test]
    async fn health_is_public_but_api_and_options_require_a_valid_token() {
        let state = Arc::new(AppState::new(
            Sha256::digest(TOKEN.as_bytes()).into(),
            vec![],
        ));
        let app = build_router(state);
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response_json(response).await, json!({"status": "ok"}));
        for method in [Method::GET, Method::OPTIONS] {
            for token in ["", "Bearer wrong", "bearer test-secret", "Bearer "] {
                let response = app
                    .clone()
                    .oneshot(
                        Request::builder()
                            .method(method.clone())
                            .uri("/v1/models")
                            .header(AUTHORIZATION, token)
                            .body(Body::empty())
                            .unwrap(),
                    )
                    .await
                    .unwrap();
                assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
                assert_eq!(response.headers()[WWW_AUTHENTICATE], "Bearer");
            }
        }
        let response = app
            .oneshot(request(Method::GET, "/v1/models", Body::empty()))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn invalid_model_bodies_are_rejected_before_contacting_an_upstream() {
        let app = build_router(Arc::new(AppState::new(
            Sha256::digest(TOKEN.as_bytes()).into(),
            vec![],
        )));
        for (body, status) in [
            ("[]", StatusCode::BAD_REQUEST),
            ("{", StatusCode::BAD_REQUEST),
            ("{}", StatusCode::BAD_REQUEST),
            (r#"{"model":null}"#, StatusCode::BAD_REQUEST),
            (r#"{"model":42}"#, StatusCode::BAD_REQUEST),
            (r#"{"model":"unknown"}"#, StatusCode::NOT_FOUND),
        ] {
            let response = app
                .clone()
                .oneshot(request(Method::POST, "/v1/chat/completions", body))
                .await
                .unwrap();
            assert_eq!(response.status(), status, "body: {body}");
            assert_eq!(
                response_json(response).await["error"]["type"],
                "invalid_request_error"
            );
        }
    }

    #[tokio::test]
    async fn chunked_uploads_are_subject_to_the_body_size_limit() {
        let app = build_router(Arc::new(AppState::new(
            Sha256::digest(TOKEN.as_bytes()).into(),
            vec![],
        )));
        let body = Body::from_stream(stream::iter([
            Ok::<_, Infallible>(Bytes::from(vec![b' '; MAX_REQUEST_BYTES / 2])),
            Ok(Bytes::from(vec![b' '; MAX_REQUEST_BYTES / 2 + 1])),
        ]));
        let response = app
            .oneshot(request(Method::POST, "/v1/completions", body))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[tokio::test]
    async fn proxy_preserves_body_query_and_status_without_forwarding_credentials() {
        let (tx, rx) = oneshot::channel();
        let tx = Arc::new(Mutex::new(Some(tx)));
        let upstream = TestServer::start(Router::new().route(
            "/v1/chat/completions",
            on(MethodFilter::PATCH, move |request: Request| {
                let tx = tx.clone();
                async move {
                    let (parts, body) = request.into_parts();
                    let body = to_bytes(body, MAX_REQUEST_BYTES).await.unwrap();
                    tx.lock()
                        .await
                        .take()
                        .unwrap()
                        .send((parts, body.clone()))
                        .unwrap();
                    Response::builder()
                        .status(StatusCode::ACCEPTED)
                        .header("content-type", "application/octet-stream")
                        .header("content-length", body.len())
                        .header("connection", "x-private")
                        .header("x-private", "not-forwarded")
                        .header("set-cookie", "a=1")
                        .header("set-cookie", "b=2")
                        .body(Body::from(body))
                        .unwrap()
                }
            }),
        ))
        .await;
        let body = Bytes::from_static(
            br#"{ "metadata": {"model": "unknown"}, "model": "rwkv7-g1k-\u0037.2b", "prompt": "hello" }"#,
        );
        let mut req = request(
            Method::PATCH,
            "/v1/chat/completions?echo=a%2Fb",
            body.clone(),
        );
        req.headers_mut()
            .insert(HOST, HeaderValue::from_static("public.example"));
        req.headers_mut()
            .insert(CONNECTION, HeaderValue::from_static("keep-alive, x-remove"));
        req.headers_mut()
            .insert("x-remove", HeaderValue::from_static("private"));
        req.headers_mut()
            .insert("x-request-id", HeaderValue::from_static("request-1"));
        let response = upstream.app().oneshot(req).await.unwrap();
        assert_eq!(response.status(), StatusCode::ACCEPTED);
        assert_eq!(response.headers()["content-length"], body.len().to_string());
        assert!(!response.headers().contains_key("x-private"));
        assert_eq!(response.headers().get_all("set-cookie").iter().count(), 2);
        assert_eq!(
            to_bytes(response.into_body(), MAX_REQUEST_BYTES)
                .await
                .unwrap(),
            body
        );
        let (parts, received) = rx.await.unwrap();
        assert_eq!(received, body);
        assert_eq!(parts.method, Method::PATCH);
        assert_eq!(parts.uri, "/v1/chat/completions?echo=a%2Fb");
        assert_eq!(
            parts.headers[HOST],
            upstream.uri.authority().unwrap().as_str()
        );
        assert_eq!(parts.headers["x-request-id"], "request-1");
        assert!(!parts.headers.contains_key(AUTHORIZATION));
        assert!(!parts.headers.contains_key("x-remove"));
    }

    #[tokio::test]
    async fn sse_is_forwarded_before_completion_without_changing_frame_contents() {
        let (tx, rx) = mpsc::channel::<Result<Bytes, Infallible>>(1);
        let receiver = Arc::new(Mutex::new(Some(rx)));
        let upstream = TestServer::start(Router::new().route(
            "/v1/chat/completions",
            post(move || {
                let receiver = receiver.clone();
                async move {
                    Response::builder()
                        .header("content-type", "text/event-stream")
                        .body(Body::from_stream(ReceiverStream::new(
                            receiver.lock().await.take().unwrap(),
                        )))
                        .unwrap()
                }
            }),
        ))
        .await;
        let first = Bytes::from_static(b"data: {\"unfinished");
        let last = Bytes::from_static(b"\":true}\n\ndata: [DONE]\n\n");
        tx.send(Ok(first.clone())).await.unwrap();
        let response = timeout(TEST_TIMEOUT, upstream.app().oneshot(completion_request()))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(response.headers()["content-type"], "text/event-stream");
        let mut body = response.into_body();
        let frame = timeout(TEST_TIMEOUT, body.frame())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(frame.into_data().unwrap(), first);
        tx.send(Ok(last.clone())).await.unwrap();
        drop(tx);
        assert_eq!(
            timeout(TEST_TIMEOUT, to_bytes(body, MAX_REQUEST_BYTES))
                .await
                .unwrap()
                .unwrap(),
            last
        );
    }

    #[tokio::test]
    async fn downstream_disconnect_releases_the_upstream_stream() {
        let (tx, rx) = oneshot::channel();
        let dropped = Arc::new(Mutex::new(Some(tx)));
        let upstream = TestServer::start(Router::new().route(
            "/v1/chat/completions",
            post(move || {
                let dropped = dropped.clone();
                async move { Body::new(PendingBody::new(dropped.lock().await.take())) }
            }),
        ))
        .await;
        let proxy = TestServer::start(upstream.app()).await;
        let client = AppState::new([0; 32], vec![]).client;
        let req = completion_request();
        let (parts, body) = req.into_parts();
        let mut req = hyper::Request::from_parts(
            parts,
            Full::new(to_bytes(body, MAX_REQUEST_BYTES).await.unwrap()),
        );
        *req.uri_mut() = format!(
            "http://{}/v1/chat/completions",
            proxy.uri.authority().unwrap()
        )
        .parse()
        .unwrap();
        let mut response = client.request(req).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        timeout(TEST_TIMEOUT, response.body_mut().frame())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        drop(response);
        timeout(TEST_TIMEOUT, rx)
            .await
            .expect("upstream stream was not released")
            .unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn more_than_1024_upstream_streams_can_remain_active_together() {
        let upstream = TestServer::start(Router::new().route(
            "/v1/chat/completions",
            post(|| async { Body::new(PendingBody::new(None)) }),
        ))
        .await;
        let app = upstream.app();
        let responses = timeout(
            Duration::from_secs(20),
            try_join_all((0..1100).map(|_| app.clone().oneshot(completion_request()))),
        )
        .await
        .expect("active streams were capped or queued")
        .unwrap();
        assert_eq!(responses.len(), 1100);
        assert!(
            responses
                .iter()
                .all(|response| response.status() == StatusCode::OK)
        );
    }

    #[tokio::test]
    async fn completed_requests_reuse_the_upstream_tcp_connection() {
        let upstream = TestServer::start(Router::new().route(
            "/v1/chat/completions",
            post(|ConnectInfo(peer): ConnectInfo<SocketAddr>| async move { peer.to_string() }),
        ))
        .await;
        let app = upstream.app();
        let mut peers = Vec::new();
        for _ in 0..2 {
            let response = app.clone().oneshot(completion_request()).await.unwrap();
            peers.push(
                to_bytes(response.into_body(), MAX_REQUEST_BYTES)
                    .await
                    .unwrap(),
            );
        }
        assert_eq!(peers[0], peers[1]);
    }

    #[tokio::test]
    async fn model_listings_are_fetched_concurrently_and_keep_configured_order() {
        let barrier = Arc::new(Barrier::new(MODEL_UPSTREAMS.len()));
        let mut servers = Vec::new();
        let mut upstreams = Vec::new();
        for &(name, _) in MODEL_UPSTREAMS {
            let barrier = barrier.clone();
            let server = TestServer::start(Router::new().route(
                "/v1/models",
                get(move || {
                    let barrier = barrier.clone();
                    async move {
                        barrier.wait().await;
                        Json(json!({"data": [{"id": name}]}))
                    }
                }),
            ))
            .await;
            upstreams.push((name, server.uri.clone()));
            servers.push(server);
        }
        let app = build_router(Arc::new(AppState::new(
            Sha256::digest(TOKEN.as_bytes()).into(),
            upstreams,
        )));
        let response = timeout(
            TEST_TIMEOUT,
            app.oneshot(request(Method::GET, "/v1/models", Body::empty())),
        )
        .await
        .expect("model listings were fetched sequentially")
        .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let expected: Vec<_> = MODEL_UPSTREAMS
            .iter()
            .map(|(name, _)| json!({"id": name}))
            .collect();
        assert_eq!(
            response_json(response).await,
            json!({"object": "list", "data": expected})
        );
    }

    #[tokio::test]
    async fn metrics_and_model_details_do_not_forward_routing_query_parameters() {
        let upstream = TestServer::start(Router::new().fallback(|req: Request| async move {
            req.uri().path_and_query().unwrap().as_str().to_owned()
        }))
        .await;
        for path in [
            format!("/metrics?model={MODEL}&extra=1"),
            format!("/v1/models/{MODEL}?extra=1"),
        ] {
            let response = upstream
                .app()
                .oneshot(request(Method::GET, &path, Body::empty()))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let body = to_bytes(response.into_body(), MAX_REQUEST_BYTES)
                .await
                .unwrap();
            assert_eq!(body, path.split('?').next().unwrap());
        }
    }

    #[tokio::test]
    async fn unavailable_upstreams_return_an_openai_style_gateway_error() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let uri = format!("http://{}", listener.local_addr().unwrap())
            .parse()
            .unwrap();
        drop(listener);
        let app = build_router(Arc::new(AppState::new(
            Sha256::digest(TOKEN.as_bytes()).into(),
            vec![(MODEL, uri)],
        )));
        let response = timeout(TEST_TIMEOUT, app.oneshot(completion_request()))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        assert_eq!(
            response_json(response).await["error"]["type"],
            "server_error"
        );
    }

    #[test]
    fn upstream_overrides_accept_only_known_models_and_http_origins() {
        let upstreams = configured_upstreams(&[format!("{MODEL}=http://127.0.0.1:8000")]).unwrap();
        assert_eq!(
            upstreams.iter().find(|(name, _)| *name == MODEL).unwrap().1,
            "http://127.0.0.1:8000"
        );
        for value in [
            "unknown=http://127.0.0.1:8000".to_owned(),
            format!("{MODEL}=https://127.0.0.1:8000"),
            format!("{MODEL}=http://user@127.0.0.1:8000"),
            format!("{MODEL}=http://127.0.0.1:8000/base"),
            format!("{MODEL}=http://127.0.0.1:8000?query=1"),
        ] {
            assert!(configured_upstreams(&[value]).is_err());
        }
    }
}
