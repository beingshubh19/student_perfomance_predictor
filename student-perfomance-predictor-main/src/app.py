import streamlit as st
import pickle
import numpy as np
import os

st.set_page_config(page_title="Student Success Advisor", page_icon="🎓")

st.title("🎓 Student Success Advisor AI")
st.write("Predict student performance based on study habits.")

# ✅ Load model safely
@st.cache_resource
def load_model():
    try:
        model_path = os.path.join("model", "model.pkl")
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("❌ model.pkl not found. Check your folder structure.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

model = load_model()

# ✅ Better UI layout
col1, col2 = st.columns(2)

with col1:
    hours = st.slider("📚 Study Hours", 0, 12, 4)
    attendance = st.slider("🏫 Attendance (%)", 0, 100, 75)

with col2:
    sleep = st.slider("😴 Sleep Hours", 0, 12, 7)
    prev = st.slider("📊 Previous Score", 0, 100, 60)

# ✅ Predict button
if st.button("🔍 Predict Performance"):
    
    input_data = np.array([[hours, attendance, sleep, prev]])

    with st.spinner("Analyzing..."):
        try:
            result = model.predict(input_data)[0]

            st.subheader(f"📈 Predicted Score: {result:.2f}")

            # ✅ Feedback system
            if result < 50:
                st.error("⚠️ High risk! Improve study habits and attendance.")
            elif result < 70:
                st.warning("📈 Average performance. You can do better!")
            else:
                st.success("🎉 Great performance! Keep it up!")

            # ✅ Input summary
            st.markdown("### 📋 Your Inputs")
            st.write(f"Study Hours: {hours}")
            st.write(f"Attendance: {attendance}%")
            st.write(f"Sleep Hours: {sleep}")
            st.write(f"Previous Score: {prev}")

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")