from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from . import crud, app 
import json
from .models import *
from django.shortcuts import render

def index(request):
    return render(request, 'index.html')

@csrf_exempt
def automate_list(request):
    if request.method == 'GET':
        # Récupérer la liste des automates
        automates = list(Automate.objects.values())
        return JsonResponse(automates, safe=False)
    elif request.method == 'POST':
        # Créer un nouvel automate
        data = json.loads(request.body)
        automate = crud.create_automate(
            data['nom'],
            data['type'],
            data['alphabet'],
            data['etats'],
            data['transitions'],
            data.get('etat_initial_id')
        )
        return JsonResponse({'id': automate.id, 'nom': automate.nom})

@csrf_exempt
def automate_detail(request, id):
    automate = crud.get_automate(id)
    if not automate:
        return JsonResponse({'error': 'Automate non trouvé'}, status=404)
    
    if request.method == 'GET':
        # Renvoyer les détails de l'automate
        etats = list(automate.etats.values())
        transitions = list(automate.transitions.values())
        return JsonResponse({
            'id': automate.id,
            'nom': automate.nom,
            'type': automate.type,
            'alphabet': automate.alphabet,
            'etat_initial_id': automate.etat_initial_id,
            'etats': etats,
            'transitions': transitions
        })
    elif request.method == 'PUT':
        # Mettre à jour l'automate
        data = json.loads(request.body)
        updated = crud.update_automate(id, **data)
        if updated:
            return JsonResponse({'status': 'success'})
        return JsonResponse({'error': 'Erreur de mise à jour'}, status=400)
    elif request.method == 'DELETE':
        # Supprimer l'automate
        if crud.delete_automate(id):
            return JsonResponse({'status': 'success'})
        return JsonResponse({'error': 'Erreur de suppression'}, status=400)

@csrf_exempt
def dessiner_automate(request, id):
    automate = crud.get_automate(id)
    if not automate:
        return JsonResponse({'error': 'Automate non trouvé'}, status=404)
    
    # Générer une représentation pour le dessin
    etats = list(automate.etats.values())
    transitions = list(automate.transitions.values())
    
    return JsonResponse({
        'etats': etats,
        'transitions': transitions,
        'etat_initial_id': automate.etat_initial_id
    })

from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import json

@csrf_exempt
def tester_mot(request, id):
    automate = crud.get_automate(id)
    if not automate:
        return JsonResponse({'error': 'Automate non trouvé'}, status=404)
    
    data = json.loads(request.body)
    mot = data.get('mot', '')
    
    # Préparer les données
    transitions = list(automate.transitions.all())
    etats = {etat.id: etat for etat in automate.etats.all()}
    etat_initial = automate.etat_initial

    def epsilon_closure(etats_set):
        """Retourne tous les états atteignables via ε-transitions depuis l'ensemble donné."""
        stack = list(etats_set)
        closure = set(etats_set)
        while stack:
            etat = stack.pop()
            for t in transitions:
                if t.source_id == etat.id and t.symbole in ('ε', ''):
                    if t.destination_id not in [e.id for e in closure]:
                        dest = etats[t.destination_id]
                        closure.add(dest)
                        stack.append(dest)
        return closure

    def avancer(etats_courants, symbole):
        """Retourne tous les états atteignables depuis etats_courants avec le symbole donné."""
        suivants = set()
        parcours = []
        for etat in etats_courants:
            for t in transitions:
                if t.source_id == etat.id and t.symbole == symbole:
                    suivant = etats[t.destination_id]
                    suivants.add(suivant)
                    parcours.append(t)
        return suivants, parcours

    # Commencer avec la fermeture epsilon de l'état initial
    courants = epsilon_closure([etat_initial])
    transitions_parcourues = []

    for c in mot:
        suivants, parcours = avancer(courants, c)
        if not suivants:
            return JsonResponse({'accepte': False, 'transitions': []})
        courants = epsilon_closure(suivants)
        transitions_parcourues.extend(parcours)

    # Vérifier si un état final est atteint
    accepte = any(etat.est_final for etat in courants)
    transitions_serialisees = [
        {
            'source': t.source.nom,
            'destination': t.destination.nom,
            'symbole': t.symbole
        }
        for t in transitions_parcourues
    ]
    return JsonResponse({
        'accepte': accepte,
        'transitions': transitions_serialisees if accepte else []
    })


@csrf_exempt
def construire_expression(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        expression = data.get('expression')
        methode = data.get('methode')
        # methode = data.get('methode', 'thompson')

        
        # Construire l'automate à partir de l'expression
        automate = app.ExpressionReguliereOperations.vers_automate(expression, methode)
        
        # Créer l'expression régulière
        expr = crud.create_expression_reguliere(expression, automate, methode)
        
        return JsonResponse({
            'expression_id': expr.id,
            'automate_id': automate.id,
            'nom': automate.nom
        })

@csrf_exempt
def langage_reconnu(request, id):
    automate = crud.get_automate(id)
    if not automate:
        return JsonResponse({'error': 'Automate non trouvé'}, status=404)
    
    # Récupérer le langage reconnu
    langage = app.AutomateOperations.langage_reconnu(automate)
    
    return JsonResponse({'langage': langage})

@csrf_exempt
def operation_automate(request, id, operation):
    automate = crud.get_automate(id)
    if not automate:
        return JsonResponse({'error': 'Automate non trouvé'}, status=404)
    
    # Appliquer l'opération
    if operation == 'determiniser':
        result = app.AutomateOperations.determiniser(automate)
    elif operation == 'minimiser':
        result = app.AutomateOperations.minimiser(automate)
    elif operation == 'completer':
        result = app.AutomateOperations.completer(automate)
    elif operation == 'complement':
        result = app.AutomateOperations.complement(automate)
    else:
        return JsonResponse({'error': 'Opération non supportée'}, status=400)
    
    return JsonResponse({
        'id': result.id,
        'nom': result.nom,
        'type': result.type
    })

@csrf_exempt
def operation_deux_automates(request, id1, id2, operation):
    automate1 = crud.get_automate(id1)
    automate2 = crud.get_automate(id2)
    if not automate1 or not automate2:
        return JsonResponse({'error': 'Automate non trouvé'}, status=404)
    
    # Appliquer l'opération
    if operation == 'reunion':
        result = app.AutomateOperations.reunion(automate1, automate2)
    elif operation == 'intersection':
        result = app.AutomateOperations.intersection(automate1, automate2)
    elif operation == 'concatenation':
        result = app.AutomateOperations.concatenation(automate1, automate2)
    else:
        return JsonResponse({'error': 'Opération non supportée'}, status=400)
    
    return JsonResponse({
        'id': result.id,
        'nom': result.nom,
        'type': result.type
    })

@csrf_exempt
def epsilon_fermeture(request, id, etat_id):
    automate = crud.get_automate(id)
    etat = Etat.objects.filter(id=etat_id).first()
    if not automate or not etat:
        return JsonResponse({'error': 'Automate ou état non trouvé'}, status=404)
    
    # Calculer l'epsilon-fermeture
    fermeture = app.AutomateOperations.epsilon_fermeture(automate, etat)
    
    return JsonResponse({
        'etat': etat_id,
        'fermeture': [e.id for e in fermeture]
    })

@csrf_exempt
def proprietes_etat(request, id, etat_id):
    automate = crud.get_automate(id)
    etat = Etat.objects.filter(id=etat_id).first()
    if not automate or not etat:
        return JsonResponse({'error': 'Automate ou état non trouvé'}, status=404)
    
    # Vérifier les propriétés
    accessibles = app.AutomateOperations.etat_accessible(automate)
    coaccessibles = app.AutomateOperations.etat_coaccessible(automate)
    utiles = app.AutomateOperations.etat_utile(automate)
    
    accessible = etat in accessibles
    coaccessible = etat in coaccessibles
    utile = etat in utiles
    
    return JsonResponse({
        'accessible': accessible,
        'coaccessible': coaccessible,
        'utile': utile
    })

@csrf_exempt
def table_transition(request, id):
    automate = crud.get_automate(id)
    if not automate:
        return JsonResponse({'error': 'Automate non trouvé'}, status=404)
    
    # Construire la table de transition
    table = []
    for transition in automate.transitions.all():
        table.append({
            'source': transition.source.nom,
            'symbole': transition.symbole,
            'destination': transition.destination.nom
        })
    
    return JsonResponse({'table': table})

def parse_equation_text(text):
    """Transforme un texte brut en dictionnaire {var: expression}"""
    lines = text.strip().splitlines()
    equations = {}
    for line in lines:
        if '=' in line:
            left, right = line.split('=', 1)
            var = left.strip()
            expr = right.strip()
            equations[var] = expr
    return equations

@csrf_exempt
def resoudre_equations_regex(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Méthode non autorisée'}, status=405)

    try:
        data = json.loads(request.body)

        # Mode 1 : format dictionnaire
        if 'equations' in data and isinstance(data['equations'], dict):
            systeme = data['equations']

        # Mode 2 : format texte brut
        elif 'texte' in data and isinstance(data['texte'], str):
            systeme = parse_equation_text(data['texte'])

        else:
            return JsonResponse({'error': 'Format invalide. Clé "equations" ou "texte" attendue.'}, status=400)

        solution = app.eliminate_and_simplify(systeme)
        return JsonResponse({'resultat': solution}, status=200)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)