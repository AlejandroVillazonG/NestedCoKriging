import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from implementations import *


st.button('Refresh')

d = 2
sup = 20

col1, col2 = st.columns(2)
with col1:
    n_1 = st.slider('Cantidad de puntos de $Y_1$', min_value=100, max_value=5_000, step=100)
with col2:
    n_2 = st.slider('Cantidad de puntos de $Y_2$', min_value=100, max_value=5_000, step=100) 

col1, col2, col3, col4 = st.columns(4)
with col1:
    nu_1 = st.number_input(label='$\\nu_1$', min_value=1e-2, value=3/2)
with col2:
    nu_2 = st.number_input(label='$\\nu_2$', min_value=1e-2, value=1/2)
with col3:
    theta_1 = st.number_input(label='$\\theta_1$', min_value=1e-2, value=1.0)
with col4:
    theta_2 = st.number_input(label='$\\theta_2$', min_value=1e-2, value=1/2)

nu_12 = ( nu_1 + nu_2 ) / 2
theta_12 = min(theta_1, theta_2)
rho_12 = (theta_12**2/(theta_1*theta_2))**(d/2) * ( gamma(nu_1+d/2)*gamma(nu_2+d/2) / (gamma(nu_1)*gamma(nu_2)) )**(1/2) * gamma(nu_12)/gamma(nu_12+d/2)

st.write('$\\nu_{12}$ = ', f'{nu_12:.2f}',
         '$\\theta_{12}$ = ', f'{theta_12:.2f}',
         '$\\rho_{12}$ = ', f'{rho_12:.2f}')
    
with st.spinner(text='Generando observaciones...'):
    X_1, X_2, Y_1, Y_2 = gen_observations(d, n_1, n_2, sup, nu_1, theta_1, nu_2, theta_2, nu_12, theta_12, rho_12)

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(8, 3))
ax1.scatter(X_1[:, 0], X_1[:, 1], c=Y_1)
ax1.set_title(r'$Y_1$')
scatter2 = ax2.scatter(X_2[:, 0], X_2[:, 1], c=Y_2)
ax2.set_title(r'$Y_2$')
fig.tight_layout()
fig.colorbar(scatter2, ax=[ax1, ax2])
st.pyplot(fig)


def generar_grilla(sqrt_n):
    xx = np.linspace(0,sup,sqrt_n)
    X, Y = np.meshgrid(xx,xx)
    return np.column_stack((X.flatten(), Y.flatten())) #Ordenados de izq a der y de abajo hacia arriba


sqrt_n = 50
X_test = generar_grilla(sqrt_n)



sigma = cov_matrix(matern_model(theta_1, nu_1), X_1, X_1)
with st.spinner(text='Calculando Kriging...'):
    Y_K = kriging(X_test, X_1, Y_1, sigma, matern_model(theta_1, nu_1))
st.title('Kriging')
fig = plt.figure(figsize=(4,2))
plt.imshow(Y_K.reshape((sqrt_n,sqrt_n)), origin='lower',
        #    interpolation='gaussian',
           extent=[0,sup,0,sup])
plt.colorbar()
st.pyplot(fig, use_container_width=False)



sigma = K(X_1, X_2,
          matern_model(theta_1, nu_1),
          matern_model(theta_2, nu_2),
          matern_model(theta_12, nu_12),
          rho_12)
with st.spinner(text='Calculando co-Kriging...'):
    Y_coK = co_kriging(X_test, X_1, X_2, Y_1, Y_2,
                       matern_model(theta_1, nu_1),
                       matern_model(theta_12, nu_12),
                       rho_12, sigma)
st.title('co-Kriging')
fig = plt.figure(figsize=(4,2))
plt.imshow(Y_coK.reshape((sqrt_n,sqrt_n)), origin='lower',
        #    interpolation='gaussian',
           extent=[0,sup,0,sup])
plt.colorbar()
st.pyplot(fig, use_container_width=False)