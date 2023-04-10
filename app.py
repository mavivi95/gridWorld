import streamlit as st
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import plotly.express as px
import plotly.graph_objects as go
import time
# Funciones 


def configBoard():
  plt.rcParams['font.family'] = 'DeJavu Serif'
  plt.rcParams['font.serif'] = ['Times New Roman']
  plt.rcParams['xtick.labelsize'] = 10
  plt.rcParams['ytick.labelsize'] = 10
  boardState = range(0,16) # Creación de estados
  actionAvailable = ['U', 'D', 'R', 'L'] # Creación de acciones disponibles, arriba, abajo, derecha e izquierda
  gameOver = {s: True if s == 0 or s == 15 else False for s in boardState} # Banderas que indican si el juego termina
  probabilitiesTransition = {(s,sNew,a): 0 for s in boardState for sNew in boardState for a in actionAvailable}
  return boardState, actionAvailable, gameOver, probabilitiesTransition

def transitionProbabilitiesSimple(boardState, actionAvailable, probabilitiesTransition):
    # Esta función calcúla las probabilidades de transición del gridworl de 4 x 4 con viento
    # Parámetros 
    # boardState = lista de estados
    # actionAvailable = lista de acciones posibles
    # probabilitiesTransition = diccionario indexado con tuplas (s', s, a) inicalizado en 0

    # Retorna 
    # probabilitiesTransition = diccionario indexado con tuplas (s', s, a) para todos los estados y todas las acciones
    
    for action in actionAvailable:
      # Acción arriba 
      if action == 'U':
        for state in boardState[1:-1]: 
          for state_new in boardState: 
              if state in [1, 2, 3]: # Análisis estados borde superior
                if state != 3:
                  probabilitiesTransition[(state, state, action)] = 0.8
                  probabilitiesTransition[(state+1, state, action)] = 0.2
                else:
                  probabilitiesTransition[(state, state, action)] = 1 # El estado 3 si va hacia arriba termina otra vez en 3
              else: # Análisis para cualquier estado no especial para esta acción
                if state_new == state - 4: # Acción arriba
                    probabilitiesTransition[(state_new, state, action)] = 0.8 
                elif state_new == state - 3: # Casilla a la derecha de la acción arriba
                    probabilitiesTransition[(state_new, state, action)] = 0.2
      # Acción abajo 
      if action == 'D':
        for state in boardState[1:-1]:
          for state_new in boardState:
              if state in [12, 13, 14]: # Análisis bordes inferiores 
                probabilitiesTransition[(state, state, action)] = 0.8
                probabilitiesTransition[(state + 1, state, action)] = 0.2
              elif state in [3, 7, 11]: # Análisis bordes derechos 
                probabilitiesTransition[(state + 4, state, action)] = 1
              else: # Análisis para cualquier estado no especial para esta acción 
                if state_new == state + 4: # Acción abajo
                    probabilitiesTransition[(state_new, state, action)] = 0.8 
                elif state_new == state + 5: # Casilla a la derecha de la acción abajo
                    probabilitiesTransition[(state_new, state, action)] = 0.2
      # Acción derecha   
      if action == 'R':
        for state in boardState[1:-1]:
          for state_new in boardState:
              if state in [3, 7, 11]: # Análisis bordes derechos 
                probabilitiesTransition[(state, state, action)] = 1
              elif state in [2, 6, 10, 14]: # Análisis columna penúltima
                probabilitiesTransition[(state+1, state, action)] = 1
              else: # Análisis para cualquier estado no especial para esta acción  
                if state_new == state + 1: # Acción derecha
                    probabilitiesTransition[(state_new, state, action)] = 0.8 
                elif state_new == state + 2: # Casilla a la derecha de la acción derecha
                    probabilitiesTransition[(state_new, state, action)] = 0.2
      # Acción izquierda 
      if action == 'L':
        for state in boardState[1:-1]:
          for state_new in boardState:
              if state in [4, 8, 12]: # Análisis bordes izquierdos
                probabilitiesTransition[(state, state, action)] = 0.8
                probabilitiesTransition[(state+1, state, action)] = 0.2
              else: # Análisis para cualquier estado no especial para esta acción 
                if state_new == state - 1: # Acción izquierda
                    probabilitiesTransition[(state_new, state, action)] = 0.8 
                elif state_new == state: # Casilla a la derecha de la acción izquierda 
                    probabilitiesTransition[(state_new, state, action)] = 0.2

    return probabilitiesTransition

def betterPolicyValueIteration(boardState, gamma, probabilitiesTransition, iteration):
    # Función de cálculo de la politica y función de valor óptimas
    # Parámetros
    # gamma = Factor de descuento 
    # probabilitiesTransition diccionario con las probabilidades de transición
    # iteration = Número de iteraciones del proceso 

    # Retorna 
    # values = Función de valor óptima para cada estado
    # policy = Política óptima para cada estado 
    # sigma = Error final proceso 
    # sigmas = Error en cada iteración max(sigma, |v - V(s)|)


    values = dict.fromkeys(boardState , 0.0) # Diccionario con valores de v(s) en función del estado 
    policy = dict.fromkeys(boardState , 0.0) # Diccionario con valores de la política en función del estado

    sigmas = [] # Errores máximos de estimación en cada iteración 
    sigma = 0 # Incluido para visualizar la iteración 0 
    # Inicia ciclo de programación dinámica 
    for k in range(iteration): 
      newValues = dict.fromkeys(boardState , 0.0)
      sigma = 0

      # Cálculo de la función de valor V(s)
      for state in boardState:
        # Para estados terminales no hay actualización 
        doneFlag = gameOver[state]
        if doneFlag:
            newValues[state] = 0
            policy[state] = 'Estado Terminal'
        else:
            valueActions = []
            actionEvaluated = []
            for action in actionAvailable:
                valueAction = 0
                for newState in boardState: # Ciclo de suma sobre todos los posibles valores de s'
                    reward = -1 # Recompensa -1 en todas las transiciones 
                    valueAction += probabilitiesTransition[(newState, state, action)]*(reward + gamma*values[newState]) 
                valueActions.append(valueAction)
                actionEvaluated.append(action)
            newValues[state] = np.max(valueActions)
        errorAbsVs = abs(values[state] - newValues[state])
        sigma= max(sigma, errorAbsVs)  
      sigmas.append(sigma)
      values = newValues
      del newValues

    # Cálculo de la política greedy respecto a V(s). 
    for state in boardState:        
      valueActionsPolicy = []
      actionEvaluatedPolicy = []
      for action in actionAvailable:
          valueActionPolicy = 0
          for newState in boardState: # Ciclo de suma sobre todos los posibles valores de s'
              reward = -1 # Recompensa -1 en todas las transiciones 
              valueActionPolicy += probabilitiesTransition[(newState, state, action)]*(reward + gamma*values[newState]) 
          valueActionsPolicy.append(valueActionPolicy)
      # En la ejecución del algoritmo se observó que numéricamente generaba diferencias 
      # para estados que en un cálculo manual estos estados eran los mismos 
      # por ejemplo el estado -3 era diferente de -3.00000004 
      # para corregir esta condición se realiza un redondeo del valor en 4 decimales 
      valueActionsPolicyRound = [round(values, 4) for values in valueActionsPolicy]
      for x in np.where(valueActionsPolicyRound == np.max(valueActionsPolicyRound)):
        policy[state] = [actionAvailable[index] for index in x.tolist()]  

    return values, policy, sigma, sigmas

def makeMatrixVs(values):
  # Función que construye una matriz para la función de valor con la estructura del gridworld 
  # Parámetros
  # values = diccionario de la función de valor 
  # Retorna 
  # matrixVs = matriz con la función de valor 
  # matrixState = matriz con los estados dispuesto en el orden del gridworld 

  matrixVs = np.zeros((4, 4))
  matrixState = np.zeros((4, 4))

  for i in boardState:
      row = (i) // 4  
      col = (i) % 4  
      matrixVs[row, col] = values.get(i, 0)
      matrixState[row, col] = i
  return matrixVs, matrixState



st.set_page_config(page_title='Bono2', layout="wide")
# diseño web

header = st.container()
st.markdown("""---""")  
context = st.container()
st.markdown("""---""")  
movement = st.container()
st.markdown("""---""")  
simulation = st.container()
st.markdown("""---""") 

with header:
  st.title( "Algoritmo iteración de valor" )
  st.header("Bono 2 ")
  st.subheader("Reinforcement Learning")
  st.write("Estudiante: Manuela Viviana Chacón Chamorro")
  

with context:
  st.header('Enunciado')
  colTex, colImage  = st.columns(2)
  with colTex:
    st.markdown('<div style="text-align: justify">Como se presenta en la Figura 1 se considera la versión con viento de un <em>GridWorld</em> de 4 x 4, lo que equivale a un total de 16 estados. En este <em>GridWorld</em>, el agente puede llevar a cabo las acciones "arriba" "U", "abajo" "D", "derecha" "R" e "izquierda" "L". Al ejecutar una acción, el agente se mueve a la casilla correspondiente con una probabilidad del 0.8, y a la casilla inmediatamente a la derecha de esta con una probabilidad del 0.2. Las casillas resaltadas del tablero indican los estados terminales. Si el agente realiza una acción que lo sacaría de la grilla, permanece en la casilla actual con una probabilidad del 0.8 y se mueve a la casilla a la derecha con una probabilidad del 0.2. Teniendo en cuenta estas dinamicas de movimiento en el gridworl se aplicará el algoritmo de iteración de valor hasta alcanzar la convergencia (aproximada), y se obtendrá la política óptima.</div>', unsafe_allow_html=True)
  with colImage:
    st.markdown('<center> <div> <img src="https://drive.google.com/uc?export=view&id=1p3mNA-eBzucpY_VE9f7q_iivaJ4DU3g8" width="250"/></div>Figura 1. <em>GridWorld</em> propuesto para el ejercicio. Los estados estados terminales estan resaltados y corresponden al estado 0 y 15. </center>', unsafe_allow_html=True)


with movement:
  st.header('Análisis de los movimientos')
  st.subheader('Movimientos convencionales')
  st.markdown('<div style="text-align: justify">En las casillas que no sean bordes el agente se mueve en la dirección deseada ("U", "D", "R", "L") con probabilidad 0.8, por acción del viento termina en la casilla a la derecha de donde hubiera quedado por la acción ejecutada con probabilidad 0.2. Considere los siguientes ejemplos:</div>', unsafe_allow_html=True)
  st.markdown('- Estado 5 y acción "U", el agente llega al estado 1 con probabilidad 0.8 y al estado 2 con probabiidad 0.2.')
  st.markdown('- Estado 6 y acción "D", el agente llega al estado 10 con probabilidad 0.8 y al estado 11 con probabiidad 0.2.')
  st.markdown('- Estado 9 y acción "R", el agente llega al estado 10 con probabilidad 0.8 y al estado 11 con probabiidad 0.2.')
  st.subheader('Movimientos bordes del tablero')
  col_1, col_2 = st.columns(2)
  with col_1:
    st.subheader('Borde superior')
    st.markdown('- Estado 1 y acción "U", el agente llega al estado 1 con probabilidad 0.8 y al estado 2 con probabilidad 0.2.  ')
    st.markdown('- Estado 2 y acción "U", el agente llega al estado 2 con probabilidad 0.8 y al estado 3 con probabilidad 0.2. ')
    st.markdown('- Estado 3 y acción "U", el agente llega al estado 3 con probabilidad 1.')
  with col_2:
    st.subheader('Borde inferior')
    st.markdown('- Estado 12 y acción "D", el agente llega al estado 12 con probabilidad 0.8 y al estado 13 con probabilidad 0.2 ')
    st.markdown('- Estado 13 y acción "D", el agente llega al estado 13 con probabilidad 0.8 y al estado 14 con probabilidad 0.2. ')
    st.markdown('- Estado 14 y acción "D", el agente llega al estado 14 con probabilidad 0.8 y al estado 15 (terminal) con probabilidad 0.2.')
  col_1, col_2 = st.columns(2)
  with col_1:
    st.subheader('Borde izquierdo')
    st.markdown('- Estado 4 y acción "L", el agente llega al estado 4 con probabilidad 0.8 y al estado 5 con probabilidad 0.2.')
    st.markdown('- Estado 8 y acción "L", el agente llega al estado 8 con probabilidad 0.8 y al estado 9 con probabilidad 0.2.')
    st.markdown('- Estado 12 y acción "L", el agente llega al estado 12 con probabilidad 0.8 y al estado 13 con probabilidad 0.2')
  with col_2: 
    st.subheader('Borde derecho')
    st.markdown('- Estado 3 y acción "R", el agente llega al estado 3 con probabilidad 1.')
    st.markdown('- Estado 7 y acción "R", el agente llega al estado 7 con probabilidad 1.')
    st.markdown('- Estado 11 y acción "R", el agente llega al estado 11 con probabilidad 1.')

with simulation:
  flag_train = False
  st.header('Simulación iteración de valor')
  with st.form(key='form1'):
    #st.header('Parámetros algoritmo')
    col_a, col_b = st.columns(2)
    with col_a:
      gamma = st.slider('Gamma', 0.0, 1.0, 1.0)
    with col_b:
      numberIteration = st.number_input("Número de iteraciones", step=1,format="%d", min_value=0)
    
    submitted = st.form_submit_button('Ejecutar')
    
    if submitted:
       flag_train = True
      # st.write(numberIteration)
      # boardState, actionAvailable, gameOver, probabilitiesTransition = configBoard()
      # probabilitiesTransition = transitionProbabilitiesSimple(boardState, actionAvailable, probabilitiesTransition)
      # for iteration in range(numberIteration):
      #   values, policy, sigma, sigmas = betterPolicyValueIteration(boardState, 1, probabilitiesTransition, iteration)
      #   matrixVs, matrixState = makeMatrixVs(values)
      #   flag_train = True
      #   # VisualizatedPolicy 

      #   figVs, axVs = plt.subplots()

      #   sns.heatmap(matrixVs, cmap= 'flare', annot = True, linewidths = 0.75, fmt='.3f', 
      #           xticklabels = False, yticklabels = False, linecolor = "white", ax = axVs)

      #   # Agregar estados en cada celda
      #   for i in range(4):
      #       for j in range(4):
      #           axVs.annotate(str(int(matrixState[i,j])), xy=(j+0.5, i+0.8), 
      #                       ha='center', va='center', color = 'black', fontsize = 12)
        
      #   # Agregar etiqueta de estado ganador
      #   for indexWin in [0, 15]:
      #     i = np.where(matrixState == indexWin)[0][0]
      #     j = np.where(matrixState == indexWin)[1][0]
      #     axVs.annotate('Game Over', xy=(j+0.5, i+0.2), 
      #                       ha='center', va='center', color = 'white', fontsize = 12)
          
      #   #axVs.set_title(r"Mapa de calor función de valor $V(s)$", fontsize = 12)

      #   figPolicy, axPolicy = plt.subplots()
      #   sns.heatmap(matrixVs, cmap= 'flare', linewidths = 0.75, ax = axPolicy,
      #           xticklabels = False, yticklabels = False, linecolor = "white")


      #   # Agregar números de estados y flechas de política 
      #   for i in range(4):
      #       for j in range(4):
      #           axPolicy.annotate(str(int(matrixState[i,j])), xy=(j+0.2, i+0.8), 
      #                       ha='center', va='center', color = 'black', fontsize = 12)
      #           winFlag = matrixState[i,j] in [0, 15]
      #           if  winFlag:
      #             pass
      #           else:
      #             # Agregar flechas que identifican la política greedy 
      #             for arrow in policy[int(matrixState[i,j])]:
      #               if arrow == 'L':
      #                 axPolicy.annotate('', xy=(j+0.2, i+0.5), xytext=(j+0.5, i+0.5), arrowprops=dict(color='white', shrink=0.05, alpha = 1))
      #               if arrow == 'R':
      #                 axPolicy.annotate('', xy=(j+0.8, i+0.5), xytext=(j+0.5, i+0.5), arrowprops=dict(color='white', shrink=0.05, alpha = 1))
      #               if arrow == 'U':
      #                 axPolicy.annotate('', xy=(j+0.5, i+0.1), xytext=(j+0.5, i+0.5), arrowprops=dict(color='white', shrink=0.05, alpha = 1))
      #               if arrow == 'D':
      #                 axPolicy.annotate('', xy=(j+0.5, i+0.9), xytext=(j+0.5, i+0.5), arrowprops=dict(color='white', shrink=0.05, alpha = 1))

      #   # Agregar etiqueta de estado ganador
      #   for indexWin in [0, 15]:
      #     i = np.where(matrixState == indexWin)[0][0]
      #     j = np.where(matrixState == indexWin)[1][0]
      #     axPolicy.annotate('Game Over', xy=(j+0.5, i+0.2), 
      #                       ha='center', va='center', color = 'white', fontsize = 12)
      #   st.write(iteration)
      #   time.sleep(1)


result_container = st.container()
st.markdown("""---""")  
with result_container:
  st.header('Resultados Algoritmo')
  
  if flag_train:
    
    for iteration in range(numberIteration+1):
      figVs, axVs = plt.subplots()
      result_figure = st.empty()
      boardState, actionAvailable, gameOver, probabilitiesTransition = configBoard()
      probabilitiesTransition = transitionProbabilitiesSimple(boardState, actionAvailable, probabilitiesTransition)
      
      values, policy, sigma, sigmas = betterPolicyValueIteration(boardState, gamma, probabilitiesTransition, iteration)
      matrixVs, matrixState = makeMatrixVs(values)
      # VisualizatedPolicy 
      
      sns.heatmap(matrixVs, cmap= 'flare', annot = True, linewidths = 0.75, fmt='.3f', 
              xticklabels = False, yticklabels = False, linecolor = "white", ax = axVs)

      # Agregar estados en cada celda
      for i in range(4):
          for j in range(4):
              axVs.annotate(str(int(matrixState[i,j])), xy=(j+0.5, i+0.8), 
                          ha='center', va='center', color = 'black', fontsize = 10)
      
      # Agregar etiqueta de estado ganador
      for indexWin in [0, 15]:
        i = np.where(matrixState == indexWin)[0][0]
        j = np.where(matrixState == indexWin)[1][0]
        axVs.annotate('Game Over', xy=(j+0.5, i+0.2), 
                          ha='center', va='center', color = 'white', fontsize = 10)
        
      #axVs.set_title(r"Mapa de calor función de valor $V(s)$", fontsize = 12)
      #st.pyplot(figVs, clear_figure = True)

      figPolicy, axPolicy = plt.subplots()
      sns.heatmap(matrixVs, cmap= 'flare', linewidths = 0.75, ax = axPolicy,
              xticklabels = False, yticklabels = False, linecolor = "white")


      # Agregar números de estados y flechas de política 
      for i in range(4):
          for j in range(4):
              axPolicy.annotate(str(int(matrixState[i,j])), xy=(j+0.2, i+0.8), 
                          ha='center', va='center', color = 'black', fontsize = 10)
              winFlag = matrixState[i,j] in [0, 15]
              if  winFlag:
                pass
              else:
                # Agregar flechas que identifican la política greedy 
                for arrow in policy[int(matrixState[i,j])]:
                  if arrow == 'L':
                    axPolicy.annotate('', xy=(j+0.2, i+0.5), xytext=(j+0.5, i+0.5), arrowprops=dict(color='white', shrink=0.05, alpha = 1))
                  if arrow == 'R':
                    axPolicy.annotate('', xy=(j+0.8, i+0.5), xytext=(j+0.5, i+0.5), arrowprops=dict(color='white', shrink=0.05, alpha = 1))
                  if arrow == 'U':
                    axPolicy.annotate('', xy=(j+0.5, i+0.1), xytext=(j+0.5, i+0.5), arrowprops=dict(color='white', shrink=0.05, alpha = 1))
                  if arrow == 'D':
                    axPolicy.annotate('', xy=(j+0.5, i+0.9), xytext=(j+0.5, i+0.5), arrowprops=dict(color='white', shrink=0.05, alpha = 1))

      # Agregar etiqueta de estado ganador
      for indexWin in [0, 15]:
        i = np.where(matrixState == indexWin)[0][0]
        j = np.where(matrixState == indexWin)[1][0]
        axPolicy.annotate('Game Over', xy=(j+0.5, i+0.2), 
                            ha='center', va='center', color = 'white', fontsize = 10)
      time.sleep(0.5)
      with result_figure:
        colVs, colPolicy = st.columns(2)
        #st.subheader(f'Iteración {iteration}')
        with colVs:
          st.write(rf'Mapa de calor función de valor $V(s)$ iteración {iteration}')
          st.pyplot(figVs, clear_figure = False)
        with colPolicy:
          st.write(rf'Política greedy respecto a $V(s)$ iteración {iteration}')
          st.pyplot(figPolicy, clear_figure = False)
      
      
      #result_figure.empty()
    
    # colVs, colPolicy = st.columns(2)
    # with colVs:
    #    st.subheader(r'Mapa de calor función de valor $V(s)$')
    #    st.pyplot(figVs, clear_figure = False)  
    # with colPolicy:
    #    st.subheader(r'Política greedy respecto a $V(s)$')
    #    st.pyplot(figPolicy, clear_figure = False) 
 
result_container_final = st.container()

with result_container_final:
  st.header('Convergencia del algoritmo')
  colVs, colPolicy = st.columns(2)
  #st.subheader(f'Iteración {iteration}')
  if flag_train:
  #   with colVs:
  #     st.subheader(rf'Mapa de calor función de valor $V(s)$ iteración {iteration}')
  #     st.pyplot(figVs, clear_figure = False)
  #   with colPolicy:
  #     st.subheader(rf'Política greedy respecto a $V(s)$ iteración {iteration}')
  #     st.pyplot(figPolicy, clear_figure = False)
    
    figSigmas, axSigmas = plt.subplots(figsize=(10, 3))
    axSigmas.plot(range(1,numberIteration+1), sigmas, color = '#AB0E6D', marker = 'o', linewidth = 2)
    axSigmas.set_title(r'Erores máximos entre las aproximaciones de $V(s)$', fontsize = 10)
    axSigmas.set_xlabel(r'Iteraciones $k$')
    #axSigmas.xticks(range(1,21))
    axSigmas.set_ylabel(r'Error $\max_{s}|(V_{k}(s)-V_{k+1}(s))|$')
    axSigmas.grid()
    st.pyplot(figSigmas, clear_figure = False)       

