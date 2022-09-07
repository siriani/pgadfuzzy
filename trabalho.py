import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Cria as variáveis do problema
# Problema: A quantidade de ração e a oferta de água as aves, influência o seu crescimento.
## Variaveis de entrada (Quantidade de ração(qtdracao) e Quantidade de água (qtdagua))
# Variável de saída (Crescimento da Ave)
qtdracao = ctrl.Antecedent(np.arange(0, 11, 1), 'Ração')
qtdagua = ctrl.Antecedent(np.arange(0, 11, 1), 'água')
crescimento = ctrl.Consequent(np.arange(0, 26, 1), 'Crescimento')

# Cria automaticamente o mapeamento entre valores nítidos e difusos 
# usando uma função de pertinência padrão (triângulo)
qtdracao.automf(names=['pouco', 'bom', 'muito'])


# Cria as funções de pertinência usando tipos variados
qtdagua['pouca'] = fuzz.trimf(qtdagua.universe, [0, 0, 5])
qtdagua['bom'] = fuzz.gaussmf(qtdagua.universe, 5, 2)
qtdagua['excelente'] = fuzz.gaussmf(qtdagua.universe, 10,3)

crescimento['baixo'] = fuzz.trimf(crescimento.universe, [0, 0, 13])
crescimento['médio'] = fuzz.trapmf(crescimento.universe, [0, 13,15, 25])
crescimento['bom'] = fuzz.trimf(crescimento.universe, [15, 25, 25])

# Visualização das funcões de pertinência
qtdracao.view()
qtdagua.view()
crescimento.view()

# Regras Fuzzy
rule1 = ctrl.Rule(qtdagua['pouca'] & qtdracao['pouco'], crescimento['baixo'])
rule2 = ctrl.Rule(qtdagua['pouca'] & qtdracao['bom'], crescimento['baixo'])
rule3 = ctrl.Rule(qtdagua['pouca'] & qtdracao['muito'], crescimento['baixo'])
rule4 = ctrl.Rule(qtdagua['bom'] & qtdracao['pouco'], crescimento['baixo'])
rule5 = ctrl.Rule(qtdagua['bom'] & qtdracao['bom'], crescimento['médio'])
rule6 = ctrl.Rule(qtdagua['bom'] & qtdracao['muito'], crescimento['médio'])
rule4 = ctrl.Rule(qtdagua['excelente'] & qtdracao['pouco'], crescimento['baixo'])
rule5 = ctrl.Rule(qtdagua['excelente'] & qtdracao['bom'], crescimento['bom'])
rule6 = ctrl.Rule(qtdagua['excelente'] & qtdracao['muito'], crescimento['bom'])

#Simulador de crescimento da ave

crescimento_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6])
crescimento_simulador = ctrl.ControlSystemSimulation(crescimento_ctrl)

# Entrando com alguns valores para quantidade de água e ração
crescimento_simulador.input['Ração'] = 3.5
crescimento_simulador.input['água'] = 9.4

# Computando o resultado
crescimento_simulador.compute()
print(crescimento_simulador.output['Crescimento'])

#Visualização do Simulador
qtdracao.view(sim=crescimento_simulador)
qtdagua.view(sim=crescimento_simulador)
crescimento.view(sim=crescimento_simulador)