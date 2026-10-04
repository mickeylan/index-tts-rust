use indextts_emotion_qwen::EmotionTextClassifier;
use std::{path::PathBuf, sync::atomic::AtomicBool};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let model_dir = PathBuf::from(
        args.next()
            .ok_or("usage: classify <qwen-model-dir> <text>")?,
    );
    let text = args
        .next()
        .ok_or("usage: classify <qwen-model-dir> <text>")?;
    let text = text.to_str().ok_or("text is not UTF-8")?;
    let mut classifier = EmotionTextClassifier::load(&model_dir)?;
    let cancelled = AtomicBool::new(false);
    let response = classifier.classify_response(text, &cancelled)?;
    let values = indextts_emotion_qwen::parse_emotion_response(text, &response);
    println!("response={response:?}\nvalues={values:?}");
    Ok(())
}
