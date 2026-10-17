use std::{
    ffi::{c_char, CStr},
    panic::{catch_unwind, AssertUnwindSafe},
    ptr,
};
use tokenizers::Tokenizer;
unsafe fn error(dst: *mut c_char, cap: usize, message: &str) {
    if !dst.is_null() && cap > 0 {
        let n = message.len().min(cap - 1);
        ptr::copy_nonoverlapping(message.as_ptr(), dst as *mut u8, n);
        *dst.add(n) = 0;
    }
}
#[no_mangle]
pub unsafe extern "C" fn eg2_tokenizer_new(
    path: *const c_char,
    err: *mut c_char,
    cap: usize,
) -> *mut Tokenizer {
    let result = catch_unwind(|| {
        if path.is_null() {
            return Err("Null tokenizer path".to_owned());
        }
        let p = CStr::from_ptr(path).to_str().map_err(|e| e.to_string())?;
        Tokenizer::from_file(p).map_err(|e| e.to_string())
    });
    match result {
        Ok(Ok(t)) => Box::into_raw(Box::new(t)),
        Ok(Err(e)) => {
            error(err, cap, &e);
            ptr::null_mut()
        }
        Err(_) => {
            error(err, cap, "Tokenizer panic");
            ptr::null_mut()
        }
    }
}
#[no_mangle]
pub unsafe extern "C" fn eg2_tokenizer_free(t: *mut Tokenizer) {
    if !t.is_null() {
        drop(Box::from_raw(t));
    }
}
// Caller owns arrays, with capacity slots per row. Outputs are zero-padded.
#[no_mangle]
pub unsafe extern "C" fn eg2_tokenizer_encode(
    t: *const Tokenizer,
    texts: *const *const c_char,
    count: usize,
    prefix: *const c_char,
    ids: *mut i64,
    mask: *mut i64,
    capacity: usize,
    lengths: *mut usize,
    err: *mut c_char,
    err_cap: usize,
) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<(), String> {
        if t.is_null() || texts.is_null() || ids.is_null() || mask.is_null() || lengths.is_null() {
            return Err("Null tokenizer argument".into());
        }
        let pre = if prefix.is_null() {
            ""
        } else {
            CStr::from_ptr(prefix).to_str().map_err(|e| e.to_string())?
        };
        for i in 0..count {
            let raw = *texts.add(i);
            if raw.is_null() {
                return Err("Null text".into());
            }
            let text = CStr::from_ptr(raw).to_str().map_err(|e| e.to_string())?;
            let input = format!("{}{}", pre, text);
            let enc = (*t).encode(input, true).map_err(|e| e.to_string())?;
            let tokens = enc.get_ids();
            if tokens.len() > capacity {
                return Err(format!(
                    "Text has {} tokens, limit {}",
                    tokens.len(),
                    capacity
                ));
            }
            *lengths.add(i) = tokens.len();
            ptr::write_bytes(ids.add(i * capacity), 0, capacity);
            ptr::write_bytes(mask.add(i * capacity), 0, capacity);
            for (j, id) in tokens.iter().enumerate() {
                *ids.add(i * capacity + j) = *id as i64;
                *mask.add(i * capacity + j) = 1;
            }
        }
        Ok(())
    }));
    match result {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => {
            error(err, err_cap, &e);
            1
        }
        Err(_) => {
            error(err, err_cap, "Tokenizer panic");
            1
        }
    }
}
