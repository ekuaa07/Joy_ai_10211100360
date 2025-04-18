import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.cluster import KMeans

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import google.generativeai as genai
from PyPDF2 import PdfReader

from transformers import pipeline

# App Configuration
st.set_page_config(page_title="AI/ML Playground", layout="wide")
st.title("🧠 AI & Machine Learning Explorer")

# Sidebar Navigation
task = st.sidebar.radio("Select a Task", ["Regression", "Clustering", "Neural Network", "LLM - Q&A"])

# ---------- Regression ----------
if task == "Regression":
    st.header("📈 Regression Task")
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv", key="regression_csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        original_df = df.copy()
        st.subheader("🔍 Dataset Preview")
        st.dataframe(df.head())

        # Handle missing values
        st.subheader("🧹 Handle Missing Values")
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                strategy = st.selectbox(
                    f"How to handle missing values in '{col}'?",
                    options=["Drop rows", "Fill with Mean", "Fill with Median", "Fill with Mode"],
                    key=col
                )
                if strategy == "Drop rows":
                    df = df.dropna(subset=[col])
                elif strategy == "Fill with Mean":
                    df[col].fillna(df[col].mean(), inplace=True)
                elif strategy == "Fill with Median":
                    df[col].fillna(df[col].median(), inplace=True)
                elif strategy == "Fill with Mode":
                    df[col].fillna(df[col].mode()[0], inplace=True)

        # Encode categorical variables
        st.subheader("🔤 Encode Categorical Columns")
        encoders = {}
        for col in df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        st.dataframe(df.head())

        # Target variable input
        st.subheader("🎯 Define the Target Variable")
        target_col = st.selectbox("Select the target column", df.columns)

        if target_col:
            X = df.drop(columns=[target_col])
            y = df[target_col]

            # Standardization
            standardize = st.checkbox("⚙️ Standardize features", value=True)
            if standardize:
                scaler = StandardScaler()
                X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

            test_size = st.slider("Test size (%)", 10, 50, 20) / 100
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.subheader("📊 Model Evaluation")
            st.write(f"**MAE**: {mean_absolute_error(y_test, y_pred):.4f}")
            st.write(f"**R² Score**: {r2_score(y_test, y_pred):.4f}")

            st.subheader("🔵 Actual vs Predicted")
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, color='teal')
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            st.pyplot(fig)

            selected_feature = st.selectbox("Feature to Visualize", X.columns)
            fig2, ax2 = plt.subplots()
            ax2.scatter(X_test[selected_feature], y_test, label="Actual")
            ax2.plot(X_test[selected_feature], y_pred, color='orange', linestyle='--', label="Predicted")
            ax2.legend()
            st.pyplot(fig2)

            st.subheader("📥 Make a Custom Prediction")
            input_data = {}
            for col in original_df.columns:
                if col != target_col:
                    if original_df[col].dtype == 'object':
                        input_data[col] = st.selectbox(col, sorted(original_df[col].dropna().unique()))
                    else:
                        input_data[col] = st.number_input(col, value=float(original_df[col].mean()))

            if st.button("Predict Value"):
                input_df = pd.DataFrame([input_data])
                for col in input_df.select_dtypes(include='object').columns:
                    if col in encoders:
                        input_df[col] = encoders[col].transform(input_df[col])

                if standardize:
                    input_df = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)

                prediction = model.predict(input_df)[0]
                st.success(f"🔮 Predicted Value: {prediction:.4f}")

# ---------- Clustering ----------
elif task == "Clustering":
    st.header("📊 Clustering Task (K-Means)")
    file = st.file_uploader("Upload dataset for clustering", type="csv", key="cluster_csv")

    if file:
        df = pd.read_csv(file)
        st.dataframe(df.head())
        features = st.multiselect("Select Features for Clustering", df.columns.tolist(), default=df.columns.tolist())
        n_clusters = st.slider("Number of Clusters", 2, 10, 3)

        model = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_data = df[features]
        model.fit(cluster_data)
        df["Cluster"] = model.labels_

        st.subheader("Clustered Data")
        st.dataframe(df)

        if len(features) == 2:
            fig, ax = plt.subplots()
            sns.scatterplot(x=features[0], y=features[1], hue="Cluster", data=df, palette="tab10", ax=ax)
            st.pyplot(fig)
        elif len(features) == 3:
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(df[features[0]], df[features[1]], df[features[2]], c=df["Cluster"])
            ax.set_xlabel(features[0])
            ax.set_ylabel(features[1])
            ax.set_zlabel(features[2])
            st.pyplot(fig)

        st.download_button("Download Clustered CSV", df.to_csv(index=False), file_name="clustered_data.csv")

# ---------- Neural Network ----------
elif task == "Neural Network":
    st.header("🔗 Neural Network Classifier")
    file = st.file_uploader("Upload classification dataset", type="csv", key="nn_csv")

    if file:
        df = pd.read_csv(file)
        st.dataframe(df.head())
        target = st.selectbox("Select Target Column", df.columns)
        X = df.drop(columns=[target])
        y = df[target]

        for col in X.select_dtypes(include=['object']).columns:
            X[col] = LabelEncoder().fit_transform(X[col])
        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)

        X = StandardScaler().fit_transform(X)

        epochs = st.slider("Epochs", 5, 100, 20)
        lr = st.slider("Learning Rate", 0.0001, 0.01, 0.001)

        model = Sequential([
            Dense(64, input_dim=X.shape[1], activation='relu'),
            Dense(32, activation='relu'),
            Dense(len(np.unique(y)), activation='softmax')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(X, y, epochs=epochs, validation_split=0.2, verbose=0)
        st.subheader("📈 Training History")
        st.line_chart(pd.DataFrame(history.history))




# # ---------- LLM ----------
# elif task == "LLM - Q&A":
#     st.header("📚 LLM Question Answering (RAG)")

#     st.markdown("Using a small open-source transformer model for Q&A on Ghana's budget.")

#     qa = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

#     with open("2025-Budget-Statement-and-Economic-Policy_v4 - Copy.pdf", "rb") as f:
#         from PyPDF2 import PdfReader
#         text = ""
#         reader = PdfReader(f)
#         for page in reader.pages:
#             text += page.extract_text()

#     question = st.text_input("Ask a question about Ghana's 2025 Budget:")

#     if question:
#         with st.spinner("Thinking..."):
#             answer = qa(question=question, context=text)
#             st.success(answer["answer"])
#             st.write("Confidence:", round(answer["score"], 4))






# # ---------- LLM ----------
# elif task == "LLM - Q&A":
#     st.header("📚 LLM Question Answering (RAG)")
    
#     # Configuration options
#     llm_choice = st.radio(
#         "Choose LLM Engine:",
#         ["Gemini (Powerful, requires API key)", "Local Model (No API needed)"],
#         horizontal=True
#     )
    
#     if llm_choice == "Gemini (Powerful, requires API key)":
#         # Gemini API configuration
#         api_key = st.text_input("Enter your Gemini API Key:", type="password")
        
#         if api_key:
#             try:
#                 import google.generativeai as genai
#                 genai.configure(api_key=api_key)
#                 model = genai.GenerativeModel('gemini-pro')
#                 st.success("✅ Gemini API connected successfully")
#             except Exception as e:
#                 st.error(f"Failed to configure Gemini: {str(e)}")
#     else:
#         # Local model configuration
#         from transformers import pipeline
#         qa = pipeline("question-answering", 
#                      model="distilbert-base-uncased-distilled-squad")
#         st.info("Using local DistilBERT model (no API required)")

#     # PDF processing (common for both approaches)
#     st.subheader("📄 Document Processing")
#     uploaded_file = st.file_uploader(
#         "Upload Ghana Budget PDF (or use default)", 
#         type="pdf",
#         accept_multiple_files=False
#     )
    
#     if uploaded_file is None:
#         # Use default file if none uploaded
#         try:
#             with open("2025-Budget-Statement-and-Economic-Policy_v4 - Copy.pdf", "rb") as f:
#                 reader = PdfReader(f)
#                 text = "".join([page.extract_text() for page in reader.pages])
#             st.info("Using default Ghana 2025 Budget document")
#         except FileNotFoundError:
#             st.error("Default PDF not found. Please upload a document.")
#             st.stop()
#     else:
#         reader = PdfReader(uploaded_file)
#         text = "".join([page.extract_text() for page in reader.pages])
#         st.success(f"Processed {len(reader.pages)} pages successfully")

#     # Question answering interface
#     st.subheader("💬 Ask Your Question")
#     question = st.text_area(
#         "Enter your question about the document:",
#         placeholder="E.g., What are the key economic priorities for 2025?"
#     )
    
#     if st.button("Get Answer") and question.strip():
#         with st.spinner("Analyzing document..."):
#             if llm_choice == "Gemini (Powerful, requires API key)" and api_key:
#                 try:
#                     prompt = f"""Use the following context to answer the question. 
#                     If the answer isn't in the context, say you don't know.
                    
#                     Context:
#                     {text[:15000]}  # Limiting context size
                    
#                     Question: {question}
                    
#                     Answer:"""
                    
#                     response = model.generate_content(prompt)
#                     st.success("Gemini's Answer:")
#                     st.write(response.text)
                    
#                 except Exception as e:
#                     st.error(f"Gemini API error: {str(e)}")
#             else:
#                 try:
#                     answer = qa(question=question, context=text[:15000])  # Limit context size
#                     st.success("Answer:")
#                     st.write(answer["answer"])
#                     st.write(f"Confidence: {answer['score']:.2%}")
#                 except Exception as e:
#                     st.error(f"Local model error: {str(e)}")

#     # Add some useful tips
#     st.markdown("""
#     **Tips for better answers:**
#     - Be specific with your questions
#     - For financial questions, include amounts or sectors
#     - The system works best with clear, direct questions
#     """)





# ---------- LLM ----------
elif task == "LLM - Q&A":
    st.header("📚 LLM Question Answering with Gemini")
    
    # Gemini API Key - MANDATORY FIRST STEP
    st.subheader("🔑 Step 1: Gemini API Configuration")
    api_key = st.text_input(
        "Enter your Gemini API Key (required):", 
        type="password",
        help="Get your key from Google AI Studio: https://aistudio.google.com/app/apikey"
    )
    
    if not api_key:
        st.warning("⚠️ Please enter your Gemini API key to continue")
        st.stop()  # Stop execution here if no API key
    
    # Initialize Gemini only after API key is provided
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-pro')
        st.success("✅ Gemini API connected successfully")
    except Exception as e:
        st.error(f"❌ Failed to configure Gemini: {str(e)}")
        st.stop()
    
    # Only show these sections AFTER API key validation
    st.subheader("📄 Step 2: Document Upload")
    uploaded_file = st.file_uploader(
        "Upload a PDF document", 
        type="pdf",
        disabled=not api_key  # Disable if no API key
    )
    
    # Process document
    text = ""
    if uploaded_file:
        try:
            reader = PdfReader(uploaded_file)
            text = "".join([page.extract_text() for page in reader.pages])
            st.success(f"📑 Processed {len(reader.pages)} pages")
        except Exception as e:
            st.error(f"Failed to process PDF: {str(e)}")
            st.stop()
    else:
        try:
            with open("2025-Budget-Statement-and-Economic-Policy_v4 - Copy.pdf", "rb") as f:
                reader = PdfReader(f)
                text = "".join([page.extract_text() for page in reader.pages])
            st.info("ℹ️ Using default Ghana Budget document")
        except FileNotFoundError:
            st.error("Default PDF not found - please upload a document")
            st.stop()
    
    # Question input (only enabled with valid API key and document)
    st.subheader("💬 Step 3: Ask Your Question")
    question = st.text_area(
        "Enter your question:",
        placeholder="What are the key economic priorities for 2025?",
        disabled=not (api_key and text)
    )
    
    if st.button("Get Answer", disabled=not (api_key and text and question.strip())):
        with st.spinner("🔍 Analyzing document..."):
            try:
                prompt = f"""Answer this question using ONLY the provided document context.
                If the answer isn't in the document, respond "This information is not in the document."

                DOCUMENT CONTEXT:
                {text[:15000]}

                QUESTION: {question}

                ANSWER:"""
                
                response = model.generate_content(prompt)
                st.success("📝 Answer:")
                st.markdown(response.text)
                
            except Exception as e:
                print("")