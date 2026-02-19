import os
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import pandas as pd
import cv2
from google import genai
import streamlit_shadcn_ui as ui


# Page Config
st.set_page_config(
    page_title="RetinAI – Research Prototype",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, sans-serif;
    }
    .footer-text {
        text-align: center;
        color: #aaa;
        font-size: 0.78rem;
        padding: 1.5rem 0 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Constants
IMG_SIZE = 224
CLASS_NAMES = [
    "CSCR (Central Serous Chorioretinopathy)",
    "Diabetic Retinopathy",
    "Disc Edema",
    "Glaucoma",
    "Healthy",
    "Macular Scar",
    "Myopia",
    "Pterygium",
    "Retinal Detachment",
    "Retinitis Pigmentosa",
]

# Model Loading
@st.cache_resource
def load_models():
    models = {"resnet": None, "densenet": None, "type": "tflite"}
    try:
        resnet_int = tf.lite.Interpreter(model_path="aug_resnet50.tflite")
        resnet_int.allocate_tensors()
        models["resnet"] = resnet_int

        densenet_int = tf.lite.Interpreter(model_path="aug_densenet201.tflite")
        densenet_int.allocate_tensors()
        models["densenet"] = densenet_int

        return models
    except Exception as e:
        st.sidebar.error(f"Model loading failed: {e}")
        return None

models_dict = load_models()

if models_dict and models_dict["type"] == "tflite":
    resnet_input_details  = models_dict["resnet"].get_input_details()
    resnet_output_details = models_dict["resnet"].get_output_details()
    dense_input_details   = models_dict["densenet"].get_input_details()
    dense_output_details  = models_dict["densenet"].get_output_details()

# Helpers
def preprocess_image(image):
    img = image.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32")
    return arr, img

def run_tflite_inference(interpreter, input_details, output_details, input_data):
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]["index"])

def predict_ensemble(arr):
    input_data = np.expand_dims(arr, axis=0)
    pred_resnet   = run_tflite_inference(models_dict["resnet"],   resnet_input_details,   resnet_output_details,   input_data)[0]
    pred_densenet = run_tflite_inference(models_dict["densenet"], dense_input_details, dense_output_details, input_data)[0]
    averaged = (pred_resnet + pred_densenet) / 2.0
    return averaged, pred_resnet, pred_densenet

def compute_occlusion_sensitivity(interpreter, input_details, output_details, img_arr, target_class_idx, progress_bar=None, start_pct=0.0):
    width, height, _ = img_arr.shape
    heatmap = np.zeros((width, height))

    input_data = np.expand_dims(img_arr, axis=0)
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    baseline = interpreter.get_tensor(output_details[0]["index"])[0][target_class_idx]

    total_steps = ((height - 32) // 32 + 1) * ((width - 32) // 32 + 1)
    step = 0

    for h in range(0, height - 32 + 1, 32):
        for w in range(0, width - 32 + 1, 32):
            step += 1
            if progress_bar:
                progress_bar.progress(min(start_pct + (step / total_steps), 1.0))

            occluded = img_arr.copy()
            occluded[h:h+32, w:w+32, :] = 0.5

            batch = np.expand_dims(occluded, axis=0)
            interpreter.set_tensor(input_details[0]["index"], batch)
            interpreter.invoke()
            prob = interpreter.get_tensor(output_details[0]["index"])[0][target_class_idx]

            heatmap[h:h+32, w:w+32] = baseline - prob

    if heatmap.max() > heatmap.min():
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    return heatmap

# Main UI
def main():
    # Sidebar 
    with st.sidebar:
        st.title("RetinAI")
        st.caption("The application is designed and developed to detect 10 eye diseases from retinal fundus images.")
        st.warning("Not a medical device. Research & educational use only.", icon="⚠️")

        st.divider()
        with st.expander("About the model"):
            st.write("Ensemble of ResNet50 and DenseNet201 trained on retinal fundus images.")
            st.write("**Reported test macro F1**")
            st.write("- ResNet50: **0.85**")
            st.write("- DenseNet201: **0.86**")
            st.divider()
            st.write("**10 detectable conditions**")
            for i, name in enumerate(CLASS_NAMES, 1):
                st.markdown(f"{i}. {name}")

    # Hero banner
    import base64
    with open("eyediseaseheader.jpg", "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    st.markdown(f"""
    <div style="position:relative; border-radius:12px; overflow:hidden; margin-bottom:1.5rem;">
        <img src="data:image/jpeg;base64,{img_b64}" style="width:100%; height:200px; object-fit:cover; display:block;">
        <div style="position:absolute; inset:0; background:linear-gradient(to right, rgba(0,0,0,0.65), rgba(0,0,0,0.1));
                    display:flex; flex-direction:column; justify-content:center; padding:2rem;">
            <h1 style="color:#fff; margin:0; font-size:1.8rem; font-family:'Inter',sans-serif; font-weight:600;">
                Eye Disease Classifier
            </h1>
            <p style="color:rgba(255,255,255,0.8); margin:0.4rem 0 0; font-size:0.95rem; font-family:'Inter',sans-serif;">
                Upload a retinal fundus image for disease prediction
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose fundus image…", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        input_arr, display_img = preprocess_image(Image.open(uploaded_file))

        col_img, col_ctrl = st.columns([5, 7], gap="medium")

        with col_img:
            image_placeholder = st.empty()
            image_placeholder.image(display_img, caption="Input image", use_container_width=True)

        with col_ctrl:
            if st.button("Run Analysis"):
                st.session_state.analyzed = True

            st.markdown("""
            **How to use**
            1. Upload a retinal fundus image (JPG/PNG)
            2. Click **Run Analysis** to classify the image
            3. View the ensemble prediction and confidence
            4. Optionally generate a heatmap to see which regions influenced the prediction
            5. Use **Interpret with AI** for an experimental explanation
            """)

            # Reset logic on file change
            if 'last_file' not in st.session_state or st.session_state.last_file != uploaded_file.name:
                for k in ['analyzed', 'animation_done', 'probs', 'probs_resnet', 'probs_densenet', 'heatmap_img', 'genai_heatmap_analysis']:
                    st.session_state.pop(k, None)
                st.session_state.last_file = uploaded_file.name

        if st.session_state.get('analyzed'):
            if models_dict is None:
                st.error("Models failed to load.")
                return

            # Inference
            if 'probs' not in st.session_state:
                with st.spinner("Running ensemble inference…"):
                    probs, p_res, p_den = predict_ensemble(input_arr)
                    st.session_state.update({
                        'probs': probs,
                        'probs_resnet': p_res,
                        'probs_densenet': p_den
                    })

            probs = st.session_state.probs
            top_idx = np.argmax(probs)
            top_class = CLASS_NAMES[top_idx]
            top_prob = probs[top_idx]

            ui.metric_card(
                title="Primary Diagnosis",
                content=top_class,
                description=f"Confidence: {top_prob:.1%}",
                key="primary_diagnosis"
            )
            # Explainability
            st.divider()
            st.subheader("Explain Prediction")
            st.caption("Occlusion Sensitivity — highlights image regions most influential for the prediction.")

            if st.button("Generate Heatmap"):
                status = st.empty()
                prog = st.progress(0)

                status.write("Step 1/2 — ResNet50")
                h_res = compute_occlusion_sensitivity(models_dict["resnet"], resnet_input_details, resnet_output_details, input_arr, top_idx, prog, 0.0)

                status.write("Step 2/2 — DenseNet201")
                h_den = compute_occlusion_sensitivity(models_dict["densenet"], dense_input_details, dense_output_details, input_arr, top_idx, prog, 0.5)

                prog.empty()
                status.success("Heatmap generated.")

                ens_h = (h_res + h_den) / 2.0
                h_u8 = np.uint8(255 * ens_h)
                h_color = cv2.applyColorMap(h_u8, cv2.COLORMAP_JET)
                h_color = cv2.cvtColor(h_color, cv2.COLOR_BGR2RGB)

                overlay = cv2.addWeighted(np.array(display_img), 0.62, h_color, 0.38, 0)
                st.session_state.heatmap_img = overlay
                st.session_state.current_file = uploaded_file.name

            if 'heatmap_img' in st.session_state and st.session_state.get('current_file') == uploaded_file.name:
                _, c1, c2, _ = st.columns([1, 3, 3, 1])
                with c1:
                    st.image(display_img, caption="Original", use_container_width=True)
                with c2:
                    st.image(st.session_state.heatmap_img, caption="Occlusion Sensitivity Heatmap", use_container_width=True)

                st.caption("Warmer colors = higher importance for the predicted class")

                # Gemini interpretation
                if st.button("Interpret with AI (Gemini)"):
                    with st.spinner("Generating interpretation…"):
                        try:
                            pil_hm = Image.fromarray(st.session_state.heatmap_img)
                            client = genai.Client()
                            prompt = f"""
                                You are an expert ophthalmologist.
                                Analyze this occlusion sensitivity heatmap overlaid on a fundus image diagnosed as **{top_class}**.
                                Red/hot areas show where the model focused most.
                                Use simple terms and briefly describe the highlighted anatomical features and whether they are consistent with known signs of {top_class}.
                                Keep response concise and factual. If you are not sure, just say you are not sure.
                                """
                            resp = client.models.generate_content(
                                model="gemini-2.5-flash",
                                contents=[pil_hm, prompt]
                            )
                            st.session_state.genai_heatmap_analysis = resp.text
                        except Exception as e:
                            st.error(f"Interpretation failed: {str(e)}")

                if 'genai_heatmap_analysis' in st.session_state:
                    st.markdown(st.session_state.genai_heatmap_analysis)
                    st.caption("Experimental — not clinically validated")

            # Model Breakdown
            st.divider()
            st.subheader("Model Breakdown")

            res_idx = np.argmax(st.session_state.probs_resnet)
            den_idx = np.argmax(st.session_state.probs_densenet)

            c1, c2, c3 = st.columns(3)
            with c1:
                ui.metric_card(
                    title="ResNet50",
                    content=f"{st.session_state.probs_resnet[res_idx]:.1%}",
                    description=CLASS_NAMES[res_idx]
                )
            with c2:
                ui.metric_card(
                    title="DenseNet201",
                    content=f"{st.session_state.probs_densenet[den_idx]:.1%}",
                    description=CLASS_NAMES[den_idx]
                )
            with c3:
                if res_idx == den_idx:
                    ui.metric_card(title="Consensus", content="Agreement", description="Both models predict the same condition")
                else:
                    ui.metric_card(title="Consensus", content="Divergence", description="Models predict different conditions")

            # Risk Profile
            st.caption("Top 5 conditions by ensemble probability")
            df = pd.DataFrame({"Condition": CLASS_NAMES, "Probability": probs})
            df = df.sort_values("Probability", ascending=False)

            for _, row in df.head(5).iterrows():
                cols = st.columns([3, 5, 1])
                cols[0].write(f"**{row['Condition']}**")
                cols[1].progress(float(row['Probability']))
                cols[2].caption(f"{row['Probability']:.1%}")

            with st.expander("Full probability distribution"):
                cmp_df = pd.DataFrame({
                    "Condition": CLASS_NAMES,
                    "ResNet50": st.session_state.probs_resnet,
                    "DenseNet201": st.session_state.probs_densenet,
                    "Ensemble": probs
                })
                st.dataframe(
                    cmp_df.style.format(precision=1, na_rep="-").format("{:.1%}", subset=["ResNet50", "DenseNet201", "Ensemble"])
                              .background_gradient(cmap="Greys", axis=None),
                    use_container_width=True
                )

    st.markdown('<div class="footer-text">Research & educational prototype · Images processed locally only</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()