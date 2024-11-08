import streamlit as st
import numpy as np
import pickle

# Função para carregar o modelo treinado
def carregar_modelo():
    with open('trained_model.sav', 'rb') as f:
        modelo = pickle.load(f)
    return modelo

# Função principal da interface
def principal():
    st.title("Classificador de Diabetes")
    st.write("Preencha as informações para saber se a pessoa tem diabetes.")

    # Entradas de dados do usuário
    gravidezes = st.number_input("Quantas vezes a pessoa foi grávida?", min_value=0, max_value=20, step=1)
    glicose = st.number_input("Qual a concentração de glicose?", min_value=0, max_value=300)
    pressao = st.number_input("Qual a pressão sanguínea?", min_value=0, max_value=200)
    espessura_pele = st.number_input("Qual a espessura da pele?", min_value=0, max_value=100)
    insulina = st.number_input("Qual o nível de insulina?", min_value=0, max_value=900)
    imc = st.number_input("Qual o IMC?", min_value=0.0, max_value=100.0, format="%.2f")
    pedigree = st.number_input("Qual o nível de pedigree de diabetes?", min_value=0.0, max_value=2.5, format="%.3f")
    idade = st.number_input("Qual a idade?", min_value=0, max_value=120)

    # Botão para classificar
    if st.button("Classificar"):
        # Organizar os dados em formato de array para o modelo
        dados = np.array([[gravidezes, glicose, pressao, espessura_pele, insulina, imc, pedigree, idade]])

        # Carregar o modelo e fazer a predição
        modelo = carregar_modelo()
        resultado = modelo.predict(dados)[0]  # O modelo retorna um valor que queremos acessar

        # Exibir o resultado
        if resultado == 1:
            st.write("A pessoa tem diabetes.")
        else:
            st.write("A pessoa não tem diabetes.")

# Executar o código principal
if __name__ == '__main__':
    principal()
