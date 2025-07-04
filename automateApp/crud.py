from .models import Automate, Etat, Transition, ExpressionReguliere

def create_automate(nom, type, alphabet, etats_data, transitions_data, etat_initial_id=None):
    automate = Automate.objects.create(nom=nom, type=type, alphabet=alphabet)
    
    # Créer les états
    etats = {}
    for etat_data in etats_data:
        etat = Etat.objects.create(automate=automate, nom=etat_data['nom'], est_final=etat_data['est_final'])
        etats[etat_data['nom']] = etat
        
    # Définir l'état initial
    if etat_initial_id:
        automate.etat_initial = etats[etat_initial_id]
        automate.save()
    
    # Créer les transitions
    for transition_data in transitions_data:
        source = etats[transition_data['source']]
        destination = etats[transition_data['destination']]
        Transition.objects.create(
            automate=automate,
            source=source,
            destination=destination,
            symbole=transition_data['symbole']
        )
    
    return automate

def get_automate(id):
    try:
        return Automate.objects.get(id=id)
    except Automate.DoesNotExist:
        return None

def update_automate(id, **kwargs):
    automate = get_automate(id)
    if automate:
        for key, value in kwargs.items():
            setattr(automate, key, value)
        automate.save()
        return automate
    return None

def delete_automate(id):
    automate = get_automate(id)
    if automate:
        automate.delete()
        return True
    return False

def create_expression_reguliere(expression, automate=None, methode_construction=None):
    expr = ExpressionReguliere.objects.create(
        expression=expression,
        automate=automate,
        methode_construction=methode_construction
    )
    return expr

def get_expression_reguliere(id):
    try:
        return ExpressionReguliere.objects.get(id=id)
    except ExpressionReguliere.DoesNotExist:
        return None

def update_expression_reguliere(id, **kwargs):
    expr = get_expression_reguliere(id)
    if expr:
        for key, value in kwargs.items():
            setattr(expr, key, value)
        expr.save()
        return expr
    return None

def delete_expression_reguliere(id):
    expr = get_expression_reguliere(id)
    if expr:
        expr.delete()
        return True
    return False