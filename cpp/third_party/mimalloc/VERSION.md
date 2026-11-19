# mimalloc — where this code comes from and why this exact version

`MI_MALLOC_VERSION 30302` (mimalloc 3.3.2).

## Origin

The source was copied from the crate used by the Rust side, not from the
system package:

```
~/.cargo/registry/src/index.crates.io-*/libmimalloc-sys-0.1.49/c_src/mimalloc/v3/
    include/  src/  LICENSE
```

## Why not the system package

Arch `extra/mimalloc` is currently 3.4.1, whereas the Rust side
(`mimalloc 0.1.52` → `libmimalloc-sys 0.1.49`) bundles 3.3.2. The study's
decision is to use **the same** memory allocator for both languages (see
`CLAUDE.md`, decision 5), so an identical C source is used rather than two
different versions.

To make sure the right branch was taken, `libmimalloc-sys/build.rs` was
checked:

```rust
let version = if env::var("CARGO_FEATURE_V2").is_ok() { "v2" } else { "v3" };
```

`rust/matmul/Cargo.toml` specifies `default-features = false` and does not
enable the `v2` feature, so Rust compiles **v3**. `cargo tree -e no-dev -f '{p} {f}'`
confirms that no features are enabled.

## Matching the compiler flags

In release mode `build.rs` passes the following defines and flags;
`cpp/build.sh` repeats them one to one:

| Flag | Why |
|---|---|
| `-O3` | the `cc` crate follows `OPT_LEVEL=3` from the release profile |
| `-DMI_DEBUG=0` | disables the heavy assertion checks |
| `-DMI_BUILD_RELEASE` | release configuration |
| `-DNDEBUG` | the same for the standard library |
| `-ftls-model=initial-exec` | faster access to thread-local variables |

`MI_MALLOC_OVERRIDE` is **not defined** on either side: Rust does not replace
the system `malloc`, it uses `#[global_allocator]`. On the C++ side the same is
done by `#include <mimalloc-new-delete.h>` in `src/main.cpp` — all
`operator new` and `operator delete` variants are redirected to mimalloc, so
`std::vector` and `std::unordered_map` allocate through it. The system `malloc`
stays untouched in both languages.

Only `src/static.c` is compiled — it includes all the other mimalloc files
itself, exactly as the Rust side does.
