from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,abort
)

from PIL import Image
import os
import sqlite3
import json
import tensorflow as tf
from werkzeug.utils import secure_filename
from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)


# ==========================================================
# FLASK APPLICATION
# ==========================================================

app = Flask(__name__)

app.secret_key = "fruitcare_ai_secret_key"


# ==========================================================
# SQLITE DATABASE CONFIGURATION
# ==========================================================

DATABASE = "farmers.db"


def get_db():

    conn = sqlite3.connect(DATABASE)

    conn.row_factory = sqlite3.Row

    return conn


# ==========================================================
# CREATE DATABASE TABLE
# ==========================================================


def init_db():
    conn = get_db()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS farmers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            phone TEXT NOT NULL,
            location TEXT NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Store fruit disease analysis history
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            farmer_id INTEGER,
            fruit TEXT NOT NULL DEFAULT 'Mango',
            disease TEXT NOT NULL,
            confidence REAL NOT NULL,
            image_filename TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (farmer_id) REFERENCES farmers(id)
        )
    """)

    conn.commit()
    conn.close()

    print("SQLite database and tables ready.")


# ==========================================================
# FILE UPLOAD CONFIGURATION
# ==========================================================

UPLOAD_FOLDER = "static/uploads"

RESULT_FOLDER = "static/results"

# Trained Mango disease classifier created by: python models.py train
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "mango_disease_model.keras")
CLASS_NAMES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "mango_class_names.json")
_cached_model = None
_cached_class_names = None


ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "webp"
}


app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

app.config["RESULT_FOLDER"] = RESULT_FOLDER


# Maximum image upload size = 10 MB

app.config["MAX_CONTENT_LENGTH"] = (
    10 * 1024 * 1024
)


# ==========================================================
# CREATE REQUIRED FOLDERS
# ==========================================================

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    RESULT_FOLDER,
    exist_ok=True
)


# ==========================================================
# HELPER FUNCTION
# ==========================================================

def allowed_file(filename):

    return (
        "." in filename
        and
        filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


def load_disease_model():
    """Load the trained model once, then reuse it for later requests."""
    global _cached_model, _cached_class_names

    if _cached_model is not None and _cached_class_names is not None:
        return _cached_model, _cached_class_names

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Run: python models.py train"
        )
    if not os.path.isfile(CLASS_NAMES_PATH):
        raise FileNotFoundError(
            f"Class-name file not found at {CLASS_NAMES_PATH}. Train the model first."
        )

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is not installed in the Python environment running Flask. "
            "Install it with: python -m pip install tensorflow pillow numpy"
        ) from exc

    _cached_model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as class_file:
        _cached_class_names = json.load(class_file)

    return _cached_model, _cached_class_names


def predict_mango_disease(image_path):
    """Return predicted disease class and model confidence for an uploaded image."""
    import numpy as np
    import tensorflow as tf

    model, class_names = load_disease_model()
    image = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    image_array = tf.keras.utils.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)

    probabilities = model.predict(image_array, verbose=0)[0]
    best_index = int(np.argmax(probabilities))
    return class_names[best_index], float(probabilities[best_index]) * 100


# ==========================================================
# INDEX PAGE
# ==========================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ==========================================================
# FARMER SIGNUP
# ==========================================================

@app.route(
    "/farmer/signup",
    methods=["GET", "POST"]
)
def farmer_signup():

    if request.method == "POST":

        # ------------------------------------------
        # GET FORM DATA
        # ------------------------------------------

        name = request.form.get(
            "name",
            ""
        ).strip()

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        phone = request.form.get(
            "phone",
            ""
        ).strip()

        location = request.form.get(
            "location",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )

        confirm_password = request.form.get(
            "confirm_password",
            ""
        )


        # ------------------------------------------
        # BASIC VALIDATION
        # ------------------------------------------

        if (
            not name
            or not email
            or not phone
            or not location
        ):

            flash(
                "Please fill in all required fields.",
                "error"
            )

            return redirect(
                url_for("farmer_signup")
            )


        # ------------------------------------------
        # PASSWORD VALIDATION
        # ------------------------------------------

        if (
            not password
            or not confirm_password
        ):

            flash(
                "Please enter your password.",
                "error"
            )

            return redirect(
                url_for("farmer_signup")
            )


        if password != confirm_password:

            flash(
                "Passwords do not match.",
                "error"
            )

            return redirect(
                url_for("farmer_signup")
            )


        if len(password) < 6:

            flash(
                "Password must contain at least 6 characters.",
                "error"
            )

            return redirect(
                url_for("farmer_signup")
            )


        # ------------------------------------------
        # DATABASE CONNECTION
        # ------------------------------------------

        conn = get_db()


        # ------------------------------------------
        # CHECK EMAIL ALREADY EXISTS
        # ------------------------------------------

        existing_farmer = conn.execute(
            """
            SELECT id
            FROM farmers
            WHERE email = ?
            """,
            (email,)
        ).fetchone()


        if existing_farmer:

            conn.close()

            flash(
                "Email already registered.",
                "error"
            )

            return redirect(
                url_for("farmer_signup")
            )


        # ------------------------------------------
        # HASH PASSWORD
        # ------------------------------------------

        hashed_password = (
            generate_password_hash(password)
        )


        # ------------------------------------------
        # INSERT FARMER INTO SQLITE
        # ------------------------------------------

        conn.execute(
            """
            INSERT INTO farmers
            (
                name,
                email,
                phone,
                location,
                password
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                name,
                email,
                phone,
                location,
                hashed_password
            )
        )


        conn.commit()

        conn.close()


        # ------------------------------------------
        # SUCCESS MESSAGE
        # ------------------------------------------

        flash(
            "Farmer account created successfully!",
            "success"
        )


        return redirect(
            url_for("farmer_login")
        )


    return render_template(
        "farmer_signup.html"
    )


# ==========================================================
# FARMER LOGIN
# ==========================================================

@app.route(
    "/farmer/login",
    methods=["GET", "POST"]
)
def farmer_login():

    if request.method == "POST":

        # ------------------------------------------
        # GET LOGIN DATA
        # ------------------------------------------

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )


        # ------------------------------------------
        # BASIC VALIDATION
        # ------------------------------------------

        if not email or not password:

            flash(
                "Please enter your email and password.",
                "error"
            )

            return redirect(
                url_for("farmer_login")
            )


        # ------------------------------------------
        # FIND FARMER FROM DATABASE
        # ------------------------------------------

        conn = get_db()

        farmer = conn.execute(
            """
            SELECT *
            FROM farmers
            WHERE email = ?
            """,
            (email,)
        ).fetchone()

        conn.close()


        # ------------------------------------------
        # CHECK EMAIL
        # ------------------------------------------

        if farmer is None:

            flash(
                "Invalid email or password.",
                "error"
            )

            return redirect(
                url_for("farmer_login")
            )


        # ------------------------------------------
        # CHECK PASSWORD
        # ------------------------------------------

        if not check_password_hash(
            farmer["password"],
            password
        ):

            flash(
                "Invalid email or password.",
                "error"
            )

            return redirect(
                url_for("farmer_login")
            )


        # ------------------------------------------
        # LOGIN SESSION
        # ------------------------------------------

        session["logged_in"] = True

        session["farmer_id"] = farmer["id"]

        session["farmer_name"] = farmer["name"]

        session["farmer_email"] = farmer["email"]

        session["farmer_phone"] = farmer["phone"]

        session["farmer_location"] = farmer["location"]


        # ------------------------------------------
        # LOGIN SUCCESS
        # ------------------------------------------

        flash(
            "Login successful!",
            "success"
        )


        return redirect(
            url_for("farmer_dashboard")
        )


    return render_template(
        "farmer_login.html"
    )


# ==========================================================
# FARMER DASHBOARD
# ==========================================================

@app.route("/farmer/dashboard")
def farmer_dashboard():

    # ------------------------------------------
    # LOGIN PROTECTION
    # ------------------------------------------

    if not session.get("logged_in"):

        flash(
            "Please login to access your dashboard.",
            "error"
        )

        return redirect(
            url_for("farmer_login")
        )


    # ------------------------------------------
    # GET FARMER ID
    # ------------------------------------------

    farmer_id = session.get(
        "farmer_id"
    )


    # ------------------------------------------
    # GET FARMER FROM DATABASE
    # ------------------------------------------

    conn = get_db()

    farmer = conn.execute(
        """
        SELECT *
        FROM farmers
        WHERE id = ?
        """,
        (farmer_id,)
    ).fetchone()

    conn.close()


    # ------------------------------------------
    # FARMER NOT FOUND
    # ------------------------------------------

    if farmer is None:

        session.clear()

        flash(
            "Farmer account not found.",
            "error"
        )

        return redirect(
            url_for("farmer_login")
        )


    # ------------------------------------------
    # SEND DATABASE DATA TO HTML
    # ------------------------------------------

    return render_template(
        "farmer_dashboard.html",

        farmer_name=farmer["name"],

        farmer_email=farmer["email"],

        farmer_phone=farmer["phone"],

        farmer_location=farmer["location"]
    )


# ==========================================================
# UPLOAD IMAGE PAGE
# ==========================================================

@app.route("/detect-disease")
def detect_disease():

    # Login protection
    if not session.get("logged_in"):
        flash(
            "Please login to upload an image.",
            "error"
        )
        return redirect(url_for("farmer_login"))

    # The page is now an image-upload page
    return render_template("upload_image.html")


# ==========================================================
# FRUIT IDENTIFICATION PAGE (UPLOAD HANDLER)
# ==========================================================


@app.route("/identify-fruit", methods=["POST"])
def identify_fruit():

    if not session.get("logged_in"):
        flash("Please login before uploading an image.", "error")
        return redirect(url_for("farmer_login"))

    if "fruit_image" not in request.files:
        flash("Please select a Mango image.", "error")
        return redirect(url_for("detect_disease"))

    file = request.files["fruit_image"]

    if not file or file.filename == "":
        flash("Please select an image.", "error")
        return redirect(url_for("detect_disease"))

    if not allowed_file(file.filename):
        flash(
            "Invalid image format. Please upload JPG, JPEG, PNG or WEBP.",
            "error"
        )
        return redirect(url_for("detect_disease"))

    import uuid

    original_filename = secure_filename(file.filename)
    extension = os.path.splitext(original_filename)[1].lower()
    filename = f"{uuid.uuid4().hex}{extension}"

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    try:
        file.save(image_path)

        prediction, confidence = predict_mango_disease(image_path)

    except (FileNotFoundError, RuntimeError) as exc:
        app.logger.error("Disease model unavailable: %s", exc)
        flash(str(exc), "error")
        return redirect(url_for("detect_disease"))

    except Exception:
        app.logger.exception("Prediction failed.")
        flash(
            "The image could not be analysed. Check the model and image, then try again.",
            "error"
        )
        return redirect(url_for("detect_disease"))

    # Save prediction history in SQLite
    conn = None

    try:
        conn = get_db()

        farmer_id = session.get("farmer_id")

        conn.execute("""
            INSERT INTO prediction_history
                (farmer_id, fruit, disease, confidence, image_filename)
            VALUES (?, ?, ?, ?, ?)
        """, (
            farmer_id,
            "Mango",
            str(prediction),
            float(confidence),
            filename
        ))

        conn.commit()

        print("SUCCESS: Mango prediction saved to database.")

    except Exception:
        app.logger.exception("DATABASE SAVE FAILED")

    finally:
        if conn is not None:
            conn.close()

    # Store image for the View More page
    session["last_uploaded_image"] = url_for(
        "static",
        filename=f"uploads/{filename}"
    )
    session["last_uploaded_filename"] = filename

    return render_template(
        "identify_fruit.html",
        image_url=url_for(
            "static",
            filename=f"uploads/{filename}"
        ),
        fruit_prediction="Mango",
        prediction=prediction,
        confidence=round(confidence, 2),
    )



# ==========================================================
# MANUAL DISEASE DETAILS (NO EXTERNAL AI API REQUIRED)
# ==========================================================

@app.route("/disease-details")
def view_disease_details():
    if not session.get("logged_in"):
        flash("Please login to view disease information.", "error")
        return redirect(url_for("farmer_login"))

    disease = request.args.get("disease", "").strip()

    disease_info = {
        "Healthy": {
            "title": "Healthy Mango",
            "description": "The model classified this image as Healthy. This means the model did not identify a known disease pattern with its highest score.",
            "causes": "A healthy mango has no obvious disease symptoms in the image. Good orchard hygiene, suitable irrigation, balanced nutrition, and careful fruit handling help maintain fruit health.",
            "symptoms": "The image may show normal fruit colour and appearance without obvious disease spots or decay.",
            "food_safety": "A healthy-looking mango can generally be eaten after washing it under clean running water and handling it hygienically. The AI result cannot guarantee food safety.",
            "prevention": "Maintain orchard hygiene, avoid fruit injuries, harvest carefully, and store fruit under suitable conditions."
        },
        "Anthracnose": {
            "title": "Mango Anthracnose",
            "description": "Anthracnose is a fungal disease commonly associated with Colletotrichum species. It can cause dark, sunken spots on mango fruit, and symptoms may become more visible during ripening.",
            "causes": "Warm, humid weather, rainfall, and prolonged moisture favour infection. The fungus can survive on infected plant material and spread through spores.",
            "symptoms": "Small dark spots may develop and enlarge into sunken brown or black lesions. Several spots can merge as the disease progresses.",
            "food_safety": "Do not eat fruit that is mouldy, rotten, or substantially decayed. Discard spoiled fruit. If a mango has only a small damaged area and is otherwise firm, clean, and not mouldy, follow trusted food-safety guidance; when in doubt, discard it. Wash the fruit before cutting.",
            "prevention": "Remove infected plant debris, improve orchard airflow, avoid unnecessary wetting of foliage and fruit, handle mangoes carefully, and use locally recommended disease management practices."
        },
        "Alternaria": {
            "title": "Mango Alternaria Disease",
            "description": "Alternaria is a group of fungi that can cause dark lesions or fruit decay in susceptible crops. The exact symptoms depend on the fungal species and growing conditions.",
            "causes": "Infection can be favoured by suitable moisture and temperature conditions, infected plant debris, and wounds on fruit.",
            "symptoms": "Possible signs include dark brown or black spots, discoloured tissue, and lesions that may expand.",
            "food_safety": "Do not eat mangoes that are mouldy, rotten, or visibly spoiled. Discard affected fruit if spoilage is present. An AI prediction alone cannot determine whether a fruit is safe to eat.",
            "prevention": "Remove diseased plant material, minimise fruit injuries, maintain orchard hygiene, and store harvested fruit appropriately."
        },
        "Black Mould Rot": {
            "title": "Mango Black Mould Rot",
            "description": "Black mould rot describes a fruit-decay condition associated with fungal growth. The precise cause should be confirmed because similar-looking symptoms can have different causes.",
            "causes": "Fungal contamination, fruit wounds, warm storage conditions, and poor handling or storage can contribute to mould growth and decay.",
            "symptoms": "Possible signs include black mould growth, dark patches, softening, and deteriorating fruit tissue.",
            "food_safety": "Do not eat mouldy or rotten mangoes. Discard visibly mouldy or decayed fruit rather than tasting it. Do not assume that cutting away a mouldy section makes the remaining fruit safe.",
            "prevention": "Handle fruit carefully, remove damaged or decayed fruit, keep storage containers clean, and store mangoes under appropriate conditions."
        },
        "Stem and Rot": {
            "title": "Mango Stem-End Rot",
            "description": "Stem-end rot is a post-harvest disease complex in which fungal infection causes decay around the stem attachment and may spread into the fruit.",
            "causes": "Disease-causing fungi may infect fruit before harvest and develop later. Warm conditions, fruit injuries, and unsuitable storage can encourage decay.",
            "symptoms": "Brown or black decay may begin near the stem end and spread inward. The fruit may soften, darken, or develop rotting tissue.",
            "food_safety": "Do not eat mangoes that have visible rot, mould, or extensive soft decay. Discard spoiled fruit. A photograph cannot establish whether the fruit is safe to consume.",
            "prevention": "Harvest carefully, avoid stem and skin injuries, remove decayed fruit, keep equipment clean, and follow suitable post-harvest storage practices."
        }
    }

    if disease not in disease_info:
        flash("Disease information was not found. Please analyse an image first.", "error")
        return redirect(url_for("detect_disease"))

    return render_template(
        "view_more.html",
        disease=disease,
        info=disease_info[disease],
        image_url=session.get("last_uploaded_image")
    )

# ==========================================================
# LEGACY DISEASE PREDICTION ROUTE
# ==========================================================

@app.route("/predict", methods=["POST"])
def predict():

    if not session.get("logged_in"):
        flash("Please login before detecting disease.", "error")
        return redirect(url_for("farmer_login"))

    if "fruit_image" not in request.files:
        flash("Please select a Mango image.", "error")
        return redirect(url_for("detect_disease"))

    file = request.files["fruit_image"]

    if not file or file.filename == "":
        flash("Please select an image.", "error")
        return redirect(url_for("detect_disease"))

    if not allowed_file(file.filename):
        flash(
            "Invalid image format. Please upload JPG, JPEG, PNG or WEBP.",
            "error"
        )
        return redirect(url_for("detect_disease"))

    import uuid

    original_filename = secure_filename(file.filename)
    extension = os.path.splitext(original_filename)[1].lower()
    filename = f"{uuid.uuid4().hex}{extension}"

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    image_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    # Run Mango disease prediction
    try:
        file.save(image_path)

        prediction, confidence = predict_mango_disease(image_path)

    except (FileNotFoundError, RuntimeError) as exc:
        app.logger.error("Disease model unavailable: %s", exc)
        flash(str(exc), "error")
        return redirect(url_for("detect_disease"))

    except Exception:
        app.logger.exception("Prediction failed.")
        flash(
            "The image could not be analysed. Check the model and image, then try again.",
            "error"
        )
        return redirect(url_for("detect_disease"))

    # Save prediction history in SQLite
    conn = get_db()

    try:
        farmer_id = session.get("farmer_id")
        print("Logged-in Farmer ID:", farmer_id)

        conn.execute("""
            INSERT INTO prediction_history
                (farmer_id, fruit, disease, confidence, image_filename)
            VALUES (?, ?, ?, ?, ?)
        """, (
            farmer_id,
            "Mango",
            str(prediction),
            float(confidence),
            filename
        ))

        conn.commit()

        print("Mango analysis saved successfully.")

    except Exception:
        app.logger.exception(
            "Failed to save Mango analysis history."
        )

    finally:
        conn.close()

    # Store the latest analysed image for the View More page
    session["last_uploaded_image"] = url_for(
        "static",
        filename=f"uploads/{filename}"
    )

    session["last_uploaded_filename"] = filename

    # Display prediction result
    return render_template(
        "detect_disease.html",
        image_url=url_for(
            "static",
            filename=f"uploads/{filename}"
        ),
        prediction=prediction,
        confidence=round(confidence, 2),
        heatmap_url=None
    )


# ==========================================================
# LOGOUT
# ==========================================================

@app.route("/logout")
def logout():

    session.clear()

    flash(
        "You have been logged out successfully.",
        "success"
    )

    return redirect(
        url_for("index")
    )


@app.route("/detect-markings")
def detect_markings():
    """Create a Grad-CAM attention overlay for the uploaded image."""
    if not session.get("logged_in"):
        flash("Please login to view disease markings.", "error")
        return redirect(url_for("farmer_login"))

    filename = secure_filename(request.args.get("image", ""))
    if not filename:
        abort(400, description="Missing image filename.")

    image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if not os.path.isfile(image_path):
        abort(404, description="Uploaded image was not found.")

    try:
        model, class_names = load_disease_model()

        # Match the image size used by the existing prediction function.
        image = tf.keras.utils.load_img(image_path, target_size=(224, 224))
        image_array = tf.keras.utils.img_to_array(image)
        input_tensor = tf.convert_to_tensor(
            np.expand_dims(image_array, axis=0), dtype=tf.float32
        )

        # Build Sequential models before accessing their tensors.
        _ = model(input_tensor, training=False)

        # Find convolutional layers, including layers inside nested models.
        conv_candidates = []
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                conv_candidates.append((layer.name, layer))
            elif isinstance(layer, tf.keras.Model):
                for nested_layer in layer.layers:
                    if isinstance(nested_layer, tf.keras.layers.Conv2D):
                        conv_candidates.append(
                            (f"{layer.name}/{nested_layer.name}", nested_layer)
                        )

        if not conv_candidates:
            raise ValueError(
                "No Conv2D layer was found. This model may not support this Grad-CAM method."
            )

        # First try a direct graph for a top-level Conv2D layer.
        grad_model = None
        last_conv_layer = None
        for _, candidate in reversed(conv_candidates):
            try:
                if "/" in _:
                    # For nested models, use the nested model's output as a safer fallback below.
                    continue
                grad_model = tf.keras.Model(
                    inputs=model.inputs,
                    outputs=[candidate.output, model.outputs[0]]
                )
                last_conv_layer = candidate
                break
            except Exception:
                grad_model = None

        if grad_model is None:
            # A robust fallback for Sequential models: make an intermediate model
            # from the model's symbolic input and the chosen convolution output.
            # If layers are nested/disconnected, report a clear diagnostic.
            raise ValueError(
                "Could not connect the convolutional layer to the model output. "
                "The model may contain a nested backbone; its layer structure must be inspected."
            )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(input_tensor, training=False)
            predicted_index_tensor = tf.argmax(predictions[0], output_type=tf.int32)
            class_score = predictions[:, predicted_index_tensor]

        gradients = tape.gradient(class_score, conv_outputs)
        if gradients is None:
            raise ValueError(
                "Gradients were not available for the selected convolutional layer."
            )

        weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
        activations = conv_outputs[0]
        heatmap = tf.reduce_sum(activations * weights, axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        heatmap = tf.math.divide_no_nan(heatmap, tf.reduce_max(heatmap))
        heatmap = heatmap.numpy()

        predicted_index = int(predicted_index_tensor.numpy())
        prediction = class_names[predicted_index]
        confidence = float(predictions[0][predicted_index].numpy()) * 100

        # Resize the heatmap to the original image dimensions and overlay it in red.
        original = Image.open(image_path).convert("RGB")
        original_width, original_height = original.size
        heatmap_image = Image.fromarray(
            np.uint8(np.clip(heatmap, 0, 1) * 255)
        ).resize((original_width, original_height), Image.Resampling.BILINEAR)
        heatmap_array = np.asarray(heatmap_image, dtype=np.float32) / 255.0

        # Keep the red overlay transparent so the fruit remains visible.
        original_array = np.asarray(original, dtype=np.float32)
        red_map = np.zeros_like(original_array)
        red_map[..., 0] = 255
        alpha = (heatmap_array ** 0.7 * 0.48)[..., None]
        overlay = np.clip(
            original_array * (1 - alpha) + red_map * alpha, 0, 255
        ).astype(np.uint8)

        os.makedirs(app.config["RESULT_FOLDER"], exist_ok=True)
        output_filename = "marked_" + os.path.splitext(filename)[0] + ".jpg"
        output_path = os.path.join(app.config["RESULT_FOLDER"], output_filename)
        Image.fromarray(overlay).save(output_path, quality=95)

        return render_template(
            "detect_markings.html",
            original_url=url_for("static", filename="uploads/" + filename),
            marked_url=url_for("static", filename="results/" + output_filename),
            prediction=prediction,
            confidence=round(confidence, 2)
        )

    except Exception as exc:
        app.logger.exception("Grad-CAM generation failed")
        abort(500, description=f"Could not generate the marking heatmap: {exc}")

@app.route("/farmer/analytics/mango")
def mango_analytics():
    return render_template(
        "mango_analyse.html",
        total_analyses=0,
        healthy_count=0,
        disease_count=0,
        avg_confidence=0,
        disease_labels=[],
        disease_values=[],
        trend_labels=[],
        trend_values=[],
        recent_analyses=[]
    )

@app.route("/farmer/history")
def farmer_history():

    if not session.get("logged_in"):
        flash("Please login to view your history.", "error")
        return redirect(url_for("farmer_login"))

    conn = get_db()

    try:
        farmer_id = session.get("farmer_id")

        history = conn.execute("""
            SELECT id, fruit, disease, confidence,
                   image_filename, created_at
            FROM prediction_history
            WHERE farmer_id = ?
            ORDER BY id DESC
        """, (farmer_id,)).fetchall()

        return render_template(
            "history.html",
            history=history
        )

    finally:
        conn.close()



from datetime import datetime

@app.route("/farmer/detection-analysis")
def detection_analysis():

    if not session.get("logged_in"):
        flash("Please login to view detection analysis.", "error")
        return redirect(url_for("farmer_login"))

    farmer_id = session.get("farmer_id")
    conn = get_db()

    try:
        total = conn.execute("""
            SELECT COUNT(*) AS total
            FROM prediction_history
            WHERE farmer_id = ?
        """, (farmer_id,)).fetchone()["total"]

        disease_rows = conn.execute("""
            SELECT disease, COUNT(*) AS total
            FROM prediction_history
            WHERE farmer_id = ?
            GROUP BY disease
            ORDER BY total DESC
        """, (farmer_id,)).fetchall()

        daily_rows = conn.execute("""
            SELECT DATE(created_at) AS day, COUNT(*) AS total
            FROM prediction_history
            WHERE farmer_id = ?
            GROUP BY DATE(created_at)
            ORDER BY day ASC
        """, (farmer_id,)).fetchall()

        avg_row = conn.execute("""
            SELECT AVG(confidence) AS average
            FROM prediction_history
            WHERE farmer_id = ?
        """, (farmer_id,)).fetchone()

        avg_confidence = round(avg_row["average"] or 0, 2)

        healthy_count = sum(
            row["total"]
            for row in disease_rows
            if str(row["disease"]).strip().lower() == "healthy"
        )

        disease_count = total - healthy_count

        return render_template(
            "detection_analysis.html",
            total=total,
            healthy_count=healthy_count,
            disease_count=disease_count,
            avg_confidence=avg_confidence,
            disease_labels=[row["disease"] for row in disease_rows],
            disease_values=[row["total"] for row in disease_rows],
            trend_labels=[row["day"] for row in daily_rows],
            trend_values=[row["total"] for row in daily_rows]
        )

    finally:
        conn.close()


# ==========================================================
# RUN APPLICATION
# ==========================================================

if __name__ == "__main__":

    # ------------------------------------------
    # CREATE DATABASE + TABLE AUTOMATICALLY
    # ------------------------------------------

    init_db()




    # ------------------------------------------
    # START FLASK SERVER
    # ------------------------------------------

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )

