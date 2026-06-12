use inferlet::{Result, runtime};

#[inferlet::main]
async fn main(input: String) -> Result<String> {
    let mut parts = input.splitn(4, ':');
    let op = parts.next().ok_or_else(|| "missing op".to_string())?;
    let namespace = parts
        .next()
        .ok_or_else(|| "missing namespace".to_string())?;
    let key = parts.next().ok_or_else(|| "missing key".to_string())?;

    match op {
        "put" => {
            let value = parts.next().ok_or_else(|| "missing value".to_string())?;
            runtime::metadata_put(namespace, key, value.as_bytes())?;
            Ok("put".to_string())
        }
        "put-big" => {
            let value = vec![0; 1024 * 1024 + 1];
            match runtime::metadata_put(namespace, key, &value) {
                Ok(()) => Ok("put".to_string()),
                Err(error) => Ok(format!("error:{error}")),
            }
        }
        "get" => match runtime::metadata_get(namespace, key)? {
            Some(value) => String::from_utf8(value).map_err(|e| e.to_string()),
            None => Ok("missing".to_string()),
        },
        "delete" => {
            let deleted = runtime::metadata_delete(namespace, key)?;
            Ok(format!("deleted:{deleted}"))
        }
        _ => Err(format!("unknown op: {op}")),
    }
}
