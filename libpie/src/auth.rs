use anyhow::{Result, anyhow};
use jsonwebtoken::{DecodingKey, EncodingKey, Header, Validation, decode, encode};
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub enum Role {
    User,
    Admin,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String, // Subject (e.g., user or service account ID)
    pub role: Role,
    pub exp: usize, // Expiration timestamp
}

static SECRET: OnceLock<String> = OnceLock::new();
pub fn init_secret(secret: &str) {
    // This will only set the secret once. Subsequent calls have no effect.
    let _ = SECRET.set(secret.to_string());
}

fn get_secret() -> Result<&'static str> {
    SECRET
        .get()
        .map(|s| s.as_str())
        .ok_or_else(|| anyhow!("JWT secret not initialized"))
}

pub fn create_jwt(user_id: &str, role: Role) -> Result<String> {
    let expiration = chrono::Utc::now()
        .checked_add_signed(chrono::Duration::days(365))
        .expect("valid timestamp")
        .timestamp();

    let claims = Claims {
        sub: user_id.to_owned(),
        role,
        exp: expiration as usize,
    };

    let secret = get_secret()?;

    encode(
        &Header::default(),
        &claims,
        &EncodingKey::from_secret(secret.as_bytes()),
    )
    .map_err(Into::into)
}

pub fn validate_jwt(token: &str) -> Result<Claims> {
    let secret = get_secret()?;

    decode::<Claims>(
        token,
        &DecodingKey::from_secret(secret.as_bytes()),
        &Validation::default(),
    )
    .map(|data| data.claims)
    .map_err(Into::into)
}






