import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

st.set_page_config (
  page_title = 'Predicción escolar',
  page_icon="🧠",
  layout = 'wide',
  initial_sidebar_state = 'expanded'
)

@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv('student_data.csv')

    # Impute missing values (as done in the notebook)
    imputer_mean = SimpleImputer(strategy='mean')
    imputer_median = SimpleImputer(strategy='median')

    # Use .loc to avoid SettingWithCopyWarning
    df.loc[:, 'sleep_hours'] = imputer_mean.fit_transform(df[['sleep_hours']])
    df.loc[:, 'family_income'] = imputer_median.fit_transform(df[['family_income']])
    df.loc[:, 'stress_level'] = imputer_median.fit_transform(df[['stress_level']])

    # One-hot encode categorical variables (as done in the notebook)
    categorical_columns = ['gender', 'private_tutoring', 'pass_fail']
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_features = encoder.fit_transform(df[categorical_columns])
    encoded_df_subset = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns), index=df.index)
    df = df.drop(columns=categorical_columns)
    df = pd.concat([df, encoded_df_subset], axis=1)

    return df

# Load the preprocessed dataframe
processed_df = load_and_preprocess_data()

#st.cache_resource.clear()

@st.cache_resource
def load_model():
    return joblib.load("student_model.pkl")

model = load_model()

# Menú dentro de la app
st.sidebar.title("Menú")
opcion = st.sidebar.radio(
    "Ir a:",
    ["Análisis exploratorio", "Predicción"]
)

# Contenido según la opción
if opcion == "Análisis exploratorio":
    st.title("👀👌 Análisis de la base de datos de los estudiantes")
    st.write("Obten la base de datos en: https://www.kaggle.com/datasets/riteshswami08/student-academic-performance-and-behavioral-factor")
    st.dataframe(processed_df.head(4))

    ####################################  

    # Calcular promedio correctamente
    calificaciones_promedio = (
        processed_df['math_score'] +
        processed_df['reading_score'] +
        processed_df['writing_score']
    ) / 3

    fig, ax = plt.subplots(figsize=(10,5))
    ax.hist(calificaciones_promedio, bins=10, edgecolor='black', alpha=0.7)
    media = np.mean(calificaciones_promedio)
    ax.axvline(media, color='red', linestyle='dashed', linewidth=2, label=f'Media: {media:.2f}')
    ax.set_xlabel('Rango de Calificaciones')
    ax.set_ylabel('Número de Estudiantes')
    ax.set_title('Distribución de Calificaciones')
    ax.legend()
    st.pyplot(fig)
    
    st.markdown("### 📊 Distribución de calificaciones")
    st.write(
        "En esta gráfica se observa la distribución del promedio de calificaciones de los estudiantes. "
        "La forma se aproxima a una campana de Gauss, lo que indica que la mayoría se concentra alrededor de un valor central. "
        "En este caso, la media es aproximadamente 62.16, lo que sugiere que muchos estudiantes logran aprobar con una calificación cercana al mínimo requerido."
    )

    st.write(
        "Además, se observa que existen menos casos en los extremos: pocos estudiantes obtienen calificaciones muy bajas o muy altas. "
        "Esto refleja una distribución equilibrada con concentración en niveles de aprobación básica dentro de la base de datos analizada."
    )

    #########################################   

    fig, ax = plt.subplots(figsize=(6,4))

    con = processed_df[processed_df['private_tutoring_True'] == 1]['math_score']
    sin = processed_df[processed_df['private_tutoring_True'] == 0]['math_score']

    ax.bar(['Con tutoría', 'Sin tutoría'], [con.mean(), sin.mean()], color=['green', 'yellow'])  
    ax.set_title("Impacto de tutoría en matemáticas")

    st.pyplot(fig)

    st.markdown("### 📈 Impacto de la tutoría")
    st.write(
        "Esta gráfica muestra el efecto de la tutoría privada en el rendimiento académico. "
        "Se observa que los estudiantes que reciben tutoría presentan, en promedio, calificaciones más altas que aquellos que no la reciben."
    )

    st.write(
        "En particular, los estudiantes con tutoría tienden a ubicarse en rangos cercanos a 70–80, "
        "mientras que los que no cuentan con este apoyo suelen estar entre 50 y 60. "
        "Esto sugiere que la tutoría tiene un impacto positivo en el desempeño académico."
    )

    #########################################

    fig, ax = plt.subplots(figsize=(6,4))

    calificaciones_promedio = (
        processed_df['math_score'] +
        processed_df['reading_score'] +
        processed_df['writing_score']
    ) / 3

    ax.set_title("Desempeño académico en contraste al nivel de estrés")
    ax.scatter(processed_df['stress_level'], calificaciones_promedio, c=processed_df['stress_level'], cmap='viridis', alpha=0.5)

    ax.set_xlabel("Estrés")
    ax.set_ylabel("Promedio académico")

    st.pyplot(fig)

    st.markdown("### 📉 Desempeño académico vs nivel de estrés")
    st.write(
        "En esta gráfica se analiza la relación entre el nivel de estrés y el promedio académico. "
        "Se observa que niveles bajos de estrés están asociados con una mayor probabilidad de obtener calificaciones altas."
    )

    st.write(
        "A medida que el estrés aumenta, las calificaciones se vuelven más dispersas. "
        "En niveles intermedios el rendimiento es más variable, y en niveles altos se observa tanto presencia de calificaciones altas como bajas, "
        "aunque con mayor dificultad para alcanzar resultados sobresalientes de forma consistente."
    )

    #########################################


elif opcion == "Predicción":
    st.title("👨‍🏫 Prediccion sobre el estado de desempeño del alumnado")

    st.subheader ('Características Personales / hábitos 👩‍💻🤟🏼')
    motivation_score = st.slider ('Puntaje de Motivación', 10, 100, 55)
    daily_study_hours = st.slider ('Horas de Estudio Diarias', 0.0, 10.0, 5.0, step=0.1)
    sleep_hours = st.slider ('Horas de Sueño', 4.0, 11.0, 7.5, step=0.1)
    stress_level = st.slider ('Nivel de Estrés', 1.0, 10.0, 5.5, step=0.1)

    st.subheader ('Características Contexto educativo 🏫⏰')
    attendance_rate = st.slider ('Tasa de Asistencia', 0.0, 1.0, 0.5, step=0.01)
    private_tutoring = st.radio('¿Recibe tutoría privada?', ('Sí', 'No'))
    private_tutoring_True = 1.0 if private_tutoring == 'Sí' else 0.0

    st.subheader ('Características Socioeconómicas 💸👨‍👩‍👦‍👦')
    family_income = st.slider ('Ingreso Familiar', 0, 2500000, 1250000)
    parental_education_level = st.slider ('Nivel Educativo de los Padres', 1, 7, 4)
    internet_quality = st.slider ('Calidad de Internet', 1, 5, 3)

    # Create a DataFrame from the inputs, ensuring correct column order and names as per X in training
    input_data = {
        'motivation_score': motivation_score,
        'daily_study_hours': daily_study_hours,
        'attendance_rate': attendance_rate,
        'family_income': family_income,
        'parental_education_level': parental_education_level,
        'internet_quality': internet_quality,
        'sleep_hours': sleep_hours,
        'stress_level': stress_level,
        'private_tutoring_True': private_tutoring_True
    }

    # Define the exact feature columns in the order the model was trained on
    feature_columns = [
        'motivation_score',
        'daily_study_hours',
        'attendance_rate',
        'family_income',
        'parental_education_level',
        'internet_quality',
        'sleep_hours',
        'stress_level',
        'private_tutoring_True'
    ]

    input_df = pd.DataFrame([input_data], columns=feature_columns)

    # Display input features
    st.subheader ('Características de entrada')
    st.write (input_df)

    # Make prediction
    if st.button('Predecir'):
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        st.subheader ('Resultado de Predicción')
        if prediction[0] == 1: # Assuming 1 is 'Pass' and 0 is 'Fail'
            st.success(f'El estudiante **aprobará** con una probabilidad de {prediction_proba[0][1]*100:.2f}% 👨🏻‍🎓')
        else:
            st.error(f'El estudiante **no aprobará** con una probabilidad de {prediction_proba[0][0]*100:.2f}% 🤷🏼')

        st.markdown("---")
        st.write("Interpretación:")
        st.info("Un resultado 'Aprobará' indica una alta probabilidad de éxito académico, mientras que 'No aprobará' sugiere que el estudiante podría enfrentar dificultades y podría necesitar apoyo adicional.")
