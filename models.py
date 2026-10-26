
from pathlib import Path
import argparse
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset" / "fruit_identification" / "Mango"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "mango_disease_model.keras"
CLASS_NAMES_PATH = MODEL_DIR / "mango_class_names.json"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
VALIDATION_SPLIT = 0.2

def load_tensorflow():
    try:
        import tensorflow as tf
        return tf
    except ImportError:
        print("TensorFlow is not installed.")
        print("Install it in your active virtual environment with:")
        print("  pip install tensorflow pillow numpy")
        raise SystemExit(1)

def validate_dataset():
    if not DATASET_DIR.is_dir():
        print(f"Dataset folder not found: {DATASET_DIR}")
        print("Update DATASET_DIR at the top of models.py to match your folder.")
        raise SystemExit(1)
    class_dirs = sorted(p for p in DATASET_DIR.iterdir() if p.is_dir())
    if len(class_dirs) < 2:
        print(f"Expected at least 2 class subfolders inside {DATASET_DIR}")
        raise SystemExit(1)
    counts = {}
    allowed = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    for folder in class_dirs:
        counts[folder.name] = sum(
            1 for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in allowed
        )
    empty = [name for name, count in counts.items() if count == 0]
    if empty:
        print("These class folders have no supported images:", ", ".join(empty))
        raise SystemExit(1)
    print("Found dataset classes:")
    for name, count in counts.items():
        print(f"  {name}: {count} images")
    return class_dirs

def train():
    tf = load_tensorflow()
    class_dirs = validate_dataset()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading images and creating training/validation sets...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=42,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=42,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )

    class_names = train_ds.class_names
    with CLASS_NAMES_PATH.open("w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3)),
        tf.keras.layers.Rescaling(1.0 / 255),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(len(class_names), activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]
    print(f"\nTraining for up to {EPOCHS} epochs...")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)
    model.save(MODEL_PATH)
    print(f"\nSaved model to: {MODEL_PATH}")
    print(f"Saved class names to: {CLASS_NAMES_PATH}")
    print("Important: review validation accuracy and test on unseen images before relying on predictions.")

def predict(image_path: str):
    tf = load_tensorflow()
    if not MODEL_PATH.exists() or not CLASS_NAMES_PATH.exists():
        print("Trained model files not found.")
        print("Run this first: python models.py train")
        raise SystemExit(1)
    img_path = Path(image_path)
    if not img_path.is_file():
        print(f"Image not found: {img_path}")
        raise SystemExit(1)
    model = tf.keras.models.load_model(MODEL_PATH)
    with CLASS_NAMES_PATH.open("r", encoding="utf-8") as f:
        class_names = json.load(f)

    img = tf.keras.utils.load_img(img_path, target_size=IMAGE_SIZE)
    arr = tf.keras.utils.img_to_array(img)
    arr = tf.expand_dims(arr, 0)
    probabilities = model.predict(arr, verbose=0)[0]
    best_idx = int(probabilities.argmax())
    print(f"Predicted Mango class: {class_names[best_idx]}")
    print(f"Model confidence: {float(probabilities[best_idx]) * 100:.2f}%")
    print("\nNote: confidence is the model's score, not a guarantee of correctness.")

def main():
    parser = argparse.ArgumentParser(description="FruitCare AI Mango disease model")
    parser.add_argument("command", choices=["train", "predict"])
    parser.add_argument("image", nargs="?", help="Image path required for predict")
    args = parser.parse_args()
    if args.command == "train":
        train()
    else:
        if not args.image:
            parser.error("predict requires an image path")
        predict(args.image)

if __name__ == "__main__":
    main()
