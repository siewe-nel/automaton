import re
from collections import defaultdict

class RegularEquationSolver:
    def __init__(self):
        self.solutions = {}
        self.dependencies = defaultdict(set)
    
    def parse_equation(self, equation_str):
        """Parse une équation sous forme de string en composants"""
        # Implémentation simplifiée - à améliorer
        terms = equation_str.split('+')
        components = {'self': '', 'rest': {}, 'const': ''}
        
        for term in terms:
            term = term.strip()
            if not term:
                continue
                
            # Auto-référence
            match = re.match(r'^([^X]*)(X\d+)$', term)
            if match:
                coeff, var = match.groups()
                if var == self.current_var:
                    components['self'] = coeff if coeff else 'ε'
                else:
                    components['rest'][var] = coeff if coeff else 'ε'
            # Terme constant
            else:
                components['const'] = term if term != 'ε' else ''
                
        return components
    
    def solve_system(self, equations):
        """Résout un système d'équations régulières"""
        self.solutions = {}
        self.equations = equations
        self.build_dependencies()
        
        # Résolution itérative
        changed = True
        while changed and len(self.solutions) < len(equations):
            changed = False
            
            for var in list(equations.keys()):
                if var not in self.solutions:
                    if self.can_solve(var):
                        self.solve_variable(var)
                        changed = True
        
        return self.solutions
    
    def build_dependencies(self):
        """Construit le graphe de dépendances entre variables"""
        self.dependencies = defaultdict(set)
        
        for var, eq in self.equations.items():
            for dep_var in eq['rest']:
                self.dependencies[var].add(dep_var)
    
    def can_solve(self, var):
        """Vérifie si une variable peut être résolue maintenant"""
        # Vérifie si toutes les dépendances sont résolues
        for dep_var in self.equations[var]['rest']:
            if dep_var not in self.solutions and dep_var != var:
                return False
        return True
    
    def solve_variable(self, var):
        """Résout une variable spécifique"""
        eq = self.equations[var]
        
        # Cas auto-référentiel
        if var in eq['rest'] or eq['self']:
            self.solve_self_referential(var)
        else:
            self.solve_non_self_referential(var)
    
    def solve_self_referential(self, var):
        """Applique le lemme d'Arden"""
        eq = self.equations[var]
        self_coeff = eq['self'] or (eq['rest'].pop(var) if var in eq['rest'] else '')
        
        # Construire le terme droit
        right_terms = []
        if eq['const']:
            right_terms.append(eq['const'])
            
        for other_var, coeff in eq['rest'].items():
            if other_var in self.solutions:
                substituted = self.multiply(coeff, self.solutions[other_var])
                right_terms.append(substituted)
            else:
                right_terms.append(self.multiply(coeff, other_var))
                
        right_side = self.add(right_terms)
        
        # Appliquer Arden
        if right_side:
            solution = self.multiply(self.star(self_coeff), right_side)
        else:
            solution = self.star(self_coeff)
            
        self.solutions[var] = solution
    
    def solve_non_self_referential(self, var):
        """Résout une variable sans auto-référence"""
        eq = self.equations[var]
        terms = []
        
        if eq['const']:
            terms.append(eq['const'])
            
        for other_var, coeff in eq['rest'].items():
            if other_var in self.solutions:
                terms.append(self.multiply(coeff, self.solutions[other_var]))
            else:
                terms.append(self.multiply(coeff, other_var))
                
        self.solutions[var] = self.add(terms) if terms else 'ε'
    
    @staticmethod
    def add(terms):
        """Combine des termes avec l'opérateur +"""
        terms = [t for t in terms if t and t != 'ε']
        if not terms:
            return 'ε'
        if len(terms) == 1:
            return terms[0]
        return '(' + '+'.join(terms) + ')'
    
    @staticmethod
    def multiply(a, b):
        """Multiplie deux termes"""
        if not a or a == 'ε':
            return b
        if not b or b == 'ε':
            return a
        return a + b
    
    @staticmethod
    def star(term):
        """Applique l'opérateur *"""
        if not term or term == 'ε':
            return 'ε'
        if len(term) == 1:
            return term + '*'
        return '(' + term + ')*'