from test_app import solve_regular_equations

# Définition du système d'équations
equations = {
        'X0': {'self': 'b', 'rest': {'X1': 'a'}, 'const': ''},
        'X1': {'self': '',  'rest': {'X2': 'a','X3':'b'}, 'const': ''},
        'X2': {'self': '',  'rest': {'X1': 'a','X3':'b'}, 'const': 'ε'},
        'X3': {'self': 'a', 'rest': {'X1': 'b'}, 'const': ''},
      }
start_var = 'X'

# Appel de la fonction
regex = solve_regular_equations(equations)
print(f"Expression régulière reconnue: {regex}")
