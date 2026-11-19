//! Data for the word frequency task.
//!
//! Unlike the matrices and the graph, the corpus is not generated
//! artificially: English-language works downloaded from Project Gutenberg
//! are used, because their word frequency distribution follows the
//! regularities of natural language. This module does not download anything
//! - it processes already downloaded .txt files, so that the list of works
//! used is explicit and can be given in the appendices.
//!
//! Processing steps:
//!   1. the Project Gutenberg licence headers and footers are removed;
//!   2. the text is normalised: upper-case letters become lower-case, all
//!      non-letter characters are treated as separators;
//!   3. the normalised text is written to a single intermediate file;
//!   4. files of the required sizes are built from it; if the corpus is not
//!      large enough, it is repeated.
//!
//! Output format - plain ASCII text: words separated by a single space, a
//! line end inserted every 16 words. A file always ends with a line end and
//! is never cut in the middle of a word.

use crate::util::{human_bytes, sha256_file, size_label, Manifest};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Result as IoResult, Write};
use std::path::{Path, PathBuf};

const WORDS_PER_LINE: usize = 16;
const CHUNK: usize = 1 << 20;

/// Removes the Project Gutenberg header and footer.
/// If the markers are absent, the whole text is returned.
fn strip_gutenberg_boilerplate(text: &str) -> &str {
    let upper = text.to_ascii_uppercase();
    let mut start = 0usize;
    let mut end = text.len();

    if let Some(pos) = upper.find("*** START OF") {
        if let Some(nl) = text[pos..].find('\n') {
            start = pos + nl + 1;
        }
    }
    if let Some(pos) = upper[start..].find("*** END OF") {
        end = start + pos;
    }
    if start >= end {
        return text;
    }
    &text[start..end]
}

/// Normalises text and appends it to the output. Returns the number of words.
/// `pending` keeps how many words have already been written on the current
/// line, so that line length stays uniform across file boundaries too.
fn normalize_into<W: Write>(text: &str, out: &mut W, pending: &mut usize) -> IoResult<u64> {
    let mut words: u64 = 0;
    let mut word: Vec<u8> = Vec::with_capacity(32);

    let flush = |word: &mut Vec<u8>, out: &mut W, pending: &mut usize| -> IoResult<bool> {
        if word.is_empty() {
            return Ok(false);
        }
        out.write_all(word)?;
        word.clear();
        *pending += 1;
        if *pending >= WORDS_PER_LINE {
            out.write_all(b"\n")?;
            *pending = 0;
        } else {
            out.write_all(b" ")?;
        }
        Ok(true)
    };

    for &b in text.as_bytes() {
        if b.is_ascii_alphabetic() {
            word.push(b.to_ascii_lowercase());
        } else if flush(&mut word, out, pending)? {
            words += 1;
        }
    }
    if flush(&mut word, out, pending)? {
        words += 1;
    }
    Ok(words)
}

struct SourceInfo {
    name: String,
    raw_bytes: u64,
    words: u64,
    sha: String,
}

/// Concatenates and normalises all input files into one intermediate file.
fn build_normalized(input_dir: &Path, tmp_path: &Path) -> IoResult<(Vec<SourceInfo>, u64)> {
    let mut files: Vec<PathBuf> = fs::read_dir(input_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && p.extension()
                    .map(|x| x.eq_ignore_ascii_case("txt"))
                    .unwrap_or(false)
        })
        .collect();
    // Sorted by name so that the result does not depend on the order
    // returned by the file system.
    files.sort();

    if files.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("no .txt files found in directory {}", input_dir.display()),
        ));
    }

    let out = File::create(tmp_path)?;
    let mut w = BufWriter::with_capacity(CHUNK, out);
    let mut infos = Vec::with_capacity(files.len());
    let mut pending = 0usize;
    let mut total_words: u64 = 0;

    for path in &files {
        let raw = fs::read(path)?;
        let text = String::from_utf8_lossy(&raw);
        let body = strip_gutenberg_boilerplate(&text);
        let words = normalize_into(body, &mut w, &mut pending)?;
        total_words += words;

        infos.push(SourceInfo {
            name: path
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default(),
            raw_bytes: raw.len() as u64,
            words,
            sha: sha256_file(path)?,
        });
        println!("    {} -> {} words", infos.last().unwrap().name, words);
    }
    if pending != 0 {
        w.write_all(b"\n")?;
    }
    w.flush()?;
    Ok((infos, total_words))
}

/// Builds a file of the given size by repeating the normalised corpus.
/// The file is cut at a word boundary.
fn write_sized(tmp_path: &Path, out_path: &Path, target: u64) -> IoResult<(u64, u32)> {
    let src_len = fs::metadata(tmp_path)?.len();
    if src_len == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "the normalised corpus is empty",
        ));
    }

    let out = File::create(out_path)?;
    let mut w = BufWriter::with_capacity(CHUNK, out);
    let mut written: u64 = 0;
    let mut repeats: u32 = 0;
    let mut buf = vec![0u8; CHUNK];

    'outer: loop {
        let f = File::open(tmp_path)?;
        let mut r = BufReader::with_capacity(CHUNK, f);
        repeats += 1;
        loop {
            let n = r.read(&mut buf)?;
            if n == 0 {
                break;
            }
            let remaining = target.saturating_sub(written);
            if (n as u64) < remaining {
                w.write_all(&buf[..n])?;
                written += n as u64;
            } else {
                // Last fragment: cut at the last word boundary.
                let take = remaining as usize;
                let slice = &buf[..take];
                let cut = slice
                    .iter()
                    .rposition(|&b| b == b' ' || b == b'\n')
                    .unwrap_or(take);
                w.write_all(&slice[..cut])?;
                w.write_all(b"\n")?;
                written += cut as u64 + 1;
                break 'outer;
            }
        }
    }
    w.flush()?;
    Ok((written, repeats))
}

/// Counts the words and the most frequent words in a file.
/// Used as an independent reference for checking the implementations.
pub fn reference_counts(path: &Path, top_n: usize) -> IoResult<(u64, Vec<(String, u64)>)> {
    let f = File::open(path)?;
    let mut r = BufReader::with_capacity(CHUNK, f);
    let mut buf = vec![0u8; CHUNK];
    let mut map: HashMap<Vec<u8>, u64> = HashMap::new();
    let mut word: Vec<u8> = Vec::with_capacity(32);
    let mut total: u64 = 0;

    loop {
        let n = r.read(&mut buf)?;
        if n == 0 {
            break;
        }
        for &b in &buf[..n] {
            if b.is_ascii_alphabetic() {
                word.push(b.to_ascii_lowercase());
            } else if !word.is_empty() {
                *map.entry(std::mem::take(&mut word)).or_insert(0) += 1;
                total += 1;
                word = Vec::with_capacity(32);
            }
        }
    }
    if !word.is_empty() {
        *map.entry(word).or_insert(0) += 1;
        total += 1;
    }

    let mut items: Vec<(String, u64)> = map
        .into_iter()
        .map(|(k, v)| (String::from_utf8_lossy(&k).to_string(), v))
        .collect();
    // Sorted by frequency descending, ties alphabetically, so that the
    // reference is unique.
    items.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    items.truncate(top_n);
    Ok((total, items))
}

pub fn generate(
    input_dir: &Path,
    sizes: &[u64],
    out_dir: &Path,
    keep_tmp: bool,
    reference: bool,
    manifest: &mut Manifest,
) -> IoResult<()> {
    manifest.section("Word frequency corpus");
    manifest.kv("Source", "Project Gutenberg, English-language works");
    manifest.kv(
        "Normalisation",
        "lower-case letters; all non-ASCII-letter characters are separators; \
         words separated by a space, a line end every 16 words",
    );

    let tmp_path = out_dir.join("corpus_normalized.tmp");
    println!("  normalising source files...");
    let (infos, total_words) = build_normalized(input_dir, &tmp_path)?;
    let norm_len = fs::metadata(&tmp_path)?.len();

    manifest.kv("Works used", &infos.len().to_string());
    manifest.kv("Words in corpus (without repetition)", &total_words.to_string());
    manifest.kv(
        "Normalised corpus size",
        &human_bytes(norm_len),
    );
    manifest.section("Works used (original files)");
    for i in &infos {
        manifest.kv(
            &i.name,
            &format!("{}, {} words, sha256={}", human_bytes(i.raw_bytes), i.words, i.sha),
        );
    }

    manifest.section("Corpus files built");
    for &target in sizes {
        let name = format!("corpus_{}.txt", size_label(target));
        let path = out_dir.join(&name);
        let (written, repeats) = write_sized(&tmp_path, &path, target)?;
        let sha = sha256_file(&path)?;

        let note = if repeats > 1 {
            format!("corpus repeated {} times", repeats)
        } else {
            "without repetition".to_string()
        };
        println!("  {}  {}  ({})", name, human_bytes(written), note);
        manifest.file_entry(&path, written, &sha, &note);

        if reference {
            println!("    computing the reference result...");
            let (total, top) = reference_counts(&path, 20)?;
            manifest.kv(&format!("  {} total words", name), &total.to_string());
            let top_str: Vec<String> = top.iter().map(|(w, c)| format!("{}={}", w, c)).collect();
            manifest.kv(&format!("  {} top 20", name), &top_str.join(", "));
        }
    }

    if !keep_tmp {
        fs::remove_file(&tmp_path)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boilerplate_removed() {
        let t = "Licence\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\nContent\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\nThe end";
        assert_eq!(strip_gutenberg_boilerplate(t).trim(), "Content");
    }

    #[test]
    fn no_markers_keeps_all() {
        let t = "just plain text";
        assert_eq!(strip_gutenberg_boilerplate(t), t);
    }

    #[test]
    fn normalization_splits_and_lowercases() {
        let mut out: Vec<u8> = Vec::new();
        let mut pending = 0;
        let n = normalize_into("Don't stop -- Now! 42x", &mut out, &mut pending).unwrap();
        let s = String::from_utf8(out).unwrap();
        assert_eq!(s, "don t stop now x ");
        assert_eq!(n, 5);
    }

    #[test]
    fn line_breaks_every_16_words() {
        let mut out: Vec<u8> = Vec::new();
        let mut pending = 0;
        let text = (0..17).map(|i| format!("w{} ", i)).collect::<String>();
        normalize_into(&text, &mut out, &mut pending).unwrap();
        let s = String::from_utf8(out).unwrap();
        assert_eq!(s.matches('\n').count(), 1);
    }
}
