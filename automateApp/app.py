from .models import Automate, Etat, Transition, ExpressionReguliere
from collections import defaultdict, deque
import re


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


class AutomateOperations:
    @staticmethod
    def determiniser(automate):
        """Déterminisation d'un AFN ou e-AFN vers AFD"""
        if automate.type == 'AFD':
            return automate
        
        # Construire la table de transitions
        alphabet = automate.alphabet.split(',')
        etats = list(automate.etats.all())
        transitions = list(automate.transitions.all())
        
        # Créer un dictionnaire des transitions
        trans_dict = defaultdict(lambda: defaultdict(set))
        for trans in transitions:
            symbole = trans.symbole if trans.symbole else 'ε'
            trans_dict[trans.source.nom][symbole].add(trans.destination.nom)
        
        # Calculer les epsilon-fermetures pour tous les états
        epsilon_closures = {}
        for etat in etats:
            epsilon_closures[etat.nom] = AutomateOperations.epsilon_fermeture(automate, etat.nom)
        
        # Construction de l'AFD
        etat_initial_closure = frozenset(epsilon_closures[automate.etat_initial.nom])
        nouveaux_etats = {etat_initial_closure}
        etats_a_traiter = [etat_initial_closure]
        transitions_afd = []
        
        while etats_a_traiter:
            etat_courant = etats_a_traiter.pop(0)
            
            for symbole in alphabet:
                if symbole.strip() == '':
                    continue
                    
                # Calculer l'ensemble des états atteignables
                destinations = set()
                for nom_etat in etat_courant:
                    if symbole in trans_dict[nom_etat]:
                        for dest in trans_dict[nom_etat][symbole]:
                            destinations.update(epsilon_closures[dest])
                
                if destinations:
                    dest_frozenset = frozenset(destinations)
                    if dest_frozenset not in nouveaux_etats:
                        nouveaux_etats.add(dest_frozenset)
                        etats_a_traiter.append(dest_frozenset)
                    
                    transitions_afd.append({
                        'source': ','.join(sorted(etat_courant)),
                        'destination': ','.join(sorted(dest_frozenset)),
                        'symbole': symbole
                    })
        
        # Créer le nouvel automate déterministe
        etats_data = []
        for etat_set in nouveaux_etats:
            nom_etat = ','.join(sorted(etat_set))
            # Un état est final s'il contient au moins un état final de l'automate original
            est_final = any(automate.etats.get(nom=nom).est_final for nom in etat_set 
                           if automate.etats.filter(nom=nom).exists())
            etats_data.append({'nom': nom_etat, 'est_final': est_final})
        
        etat_initial_nom = ','.join(sorted(etat_initial_closure))
        
        return create_automate(
            nom=f"{automate.nom}_determinise",
            type='AFD',
            alphabet=automate.alphabet,
            etats_data=etats_data,
            transitions_data=transitions_afd,
            etat_initial_id=etat_initial_nom
        )
    
    @staticmethod
    def minimiser(automate):
        """Minimisation d'un AFD par l'algorithme de Hopcroft"""
        if automate.type != 'AFD':
            automate = AutomateOperations.determiniser(automate)
            
        # Supprimer les états inutiles
        etats_utiles = AutomateOperations.etat_utile(automate)
        if len(etats_utiles) != automate.etats.count():
            # Reconstruire l'automate avec seulement les états utiles
            etats_utiles_noms = [e.nom for e in etats_utiles]
            etats_data = [{'nom': e.nom, 'est_final': e.est_final} for e in etats_utiles]
            transitions_data = []
            for trans in automate.transitions.all():
                if trans.source.nom in etats_utiles_noms and trans.destination.nom in etats_utiles_noms:
                    transitions_data.append({
                        'source': trans.source.nom,
                        'destination': trans.destination.nom,
                        'symbole': trans.symbole
                    })
            
            automate = create_automate(
                nom=f"{automate.nom}_clean",
                type='AFD',
                alphabet=automate.alphabet,
                etats_data=etats_data,
                transitions_data=transitions_data,
                etat_initial_id=automate.etat_initial.nom if automate.etat_initial.nom in etats_utiles_noms else None
            )
        
        # Algorithme de minimisation par classes d'équivalence
        etats = list(automate.etats.all())
        alphabet = [s.strip() for s in automate.alphabet.split(',') if s.strip()]
        
        # Partition initiale : états finaux et non-finaux
        finaux = [e for e in etats if e.est_final]
        non_finaux = [e for e in etats if not e.est_final]
        
        partitions = []
        if finaux:
            partitions.append(finaux)
        if non_finaux:
            partitions.append(non_finaux)
        
        # Construire la table de transitions
        trans_dict = {}
        for etat in etats:
            trans_dict[etat.nom] = {}
            for symbole in alphabet:
                trans_dict[etat.nom][symbole] = None
        
        for trans in automate.transitions.all():
            if trans.symbole in alphabet:
                trans_dict[trans.source.nom][trans.symbole] = trans.destination.nom
        
        # Raffiner les partitions
        changed = True
        while changed:
            changed = False
            nouvelles_partitions = []
            
            for partition in partitions:
                # Grouper les états par leur comportement
                groupes = defaultdict(list)
                for etat in partition:
                    signature = []
                    for symbole in alphabet:
                        dest = trans_dict[etat.nom][symbole]
                        # Trouver dans quelle partition se trouve la destination
                        dest_partition = -1
                        if dest:
                            for i, p in enumerate(partitions):
                                if any(e.nom == dest for e in p):
                                    dest_partition = i
                                    break
                        signature.append(dest_partition)
                    groupes[tuple(signature)].append(etat)
                
                # Si le groupe a été divisé, marquer le changement
                if len(groupes) > 1:
                    changed = True
                
                nouvelles_partitions.extend(groupes.values())
            
            partitions = nouvelles_partitions
        
        # Construire l'automate minimisé
        etats_data = []
        etat_representants = {}
        
        for i, partition in enumerate(partitions):
            # Choisir un représentant pour chaque partition
            representant = partition[0]
            nom_classe = f"q{i}"
            etats_data.append({
                'nom': nom_classe,
                'est_final': representant.est_final
            })
            
            # Mapper tous les états de la partition vers le représentant
            for etat in partition:
                etat_representants[etat.nom] = nom_classe
        
        # Construire les transitions
        transitions_data = []
        transitions_ajoutees = set()
        
        for trans in automate.transitions.all():
            if trans.symbole in alphabet:
                source_classe = etat_representants[trans.source.nom]
                dest_classe = etat_representants[trans.destination.nom]
                
                transition_key = (source_classe, dest_classe, trans.symbole)
                if transition_key not in transitions_ajoutees:
                    transitions_data.append({
                        'source': source_classe,
                        'destination': dest_classe,
                        'symbole': trans.symbole
                    })
                    transitions_ajoutees.add(transition_key)
        
        # Déterminer l'état initial
        etat_initial_classe = etat_representants[automate.etat_initial.nom]
        
        return create_automate(
            nom=f"{automate.nom}_minimise",
            type='AFD',
            alphabet=automate.alphabet,
            etats_data=etats_data,
            transitions_data=transitions_data,
            etat_initial_id=etat_initial_classe
        )
    
    @staticmethod
    def completer(automate):
        """Rendre un automate complet en ajoutant un état puits si nécessaire"""
        if automate.type != 'AFD':
            automate = AutomateOperations.determiniser(automate)
        
        alphabet = [s.strip() for s in automate.alphabet.split(',') if s.strip()]
        etats = list(automate.etats.all())
        transitions = list(automate.transitions.all())
        
        # Vérifier si l'automate est déjà complet
        trans_dict = defaultdict(dict)
        for trans in transitions:
            trans_dict[trans.source.nom][trans.symbole] = trans.destination.nom
        
        manque_transitions = False
        for etat in etats:
            for symbole in alphabet:
                if symbole not in trans_dict[etat.nom]:
                    manque_transitions = True
                    break
            if manque_transitions:
                break
        
        if not manque_transitions:
            return automate
        
        # Ajouter un état puits
        etats_data = [{'nom': e.nom, 'est_final': e.est_final} for e in etats]
        etats_data.append({'nom': 'puits', 'est_final': False})
        
        transitions_data = []
        for trans in transitions:
            transitions_data.append({
                'source': trans.source.nom,
                'destination': trans.destination.nom,
                'symbole': trans.symbole
            })
        
        # Ajouter les transitions manquantes vers l'état puits
        for etat in etats:
            for symbole in alphabet:
                if symbole not in trans_dict[etat.nom]:
                    transitions_data.append({
                        'source': etat.nom,
                        'destination': 'puits',
                        'symbole': symbole
                    })
        
        # Ajouter les transitions de l'état puits vers lui-même
        for symbole in alphabet:
            transitions_data.append({
                'source': 'puits',
                'destination': 'puits',
                'symbole': symbole
            })
        
        return create_automate(
            nom=f"{automate.nom}_complet",
            type='AFD',
            alphabet=automate.alphabet,
            etats_data=etats_data,
            transitions_data=transitions_data,
            etat_initial_id=automate.etat_initial.nom
        )
    
    @staticmethod
    def complement(automate):
        """Complémentation d'un AFD complet"""
        # S'assurer que l'automate est un AFD complet
        automate = AutomateOperations.completer(automate)
        
        # Inverser les états finaux et non-finaux
        etats_data = []
        for etat in automate.etats.all():
            etats_data.append({
                'nom': etat.nom,
                'est_final': not etat.est_final
            })
        
        transitions_data = []
        for trans in automate.transitions.all():
            transitions_data.append({
                'source': trans.source.nom,
                'destination': trans.destination.nom,
                'symbole': trans.symbole
            })
        
        return create_automate(
            nom=f"complement_{automate.nom}",
            type='AFD',
            alphabet=automate.alphabet,
            etats_data=etats_data,
            transitions_data=transitions_data,
            etat_initial_id=automate.etat_initial.nom
        )
    
    @staticmethod
    def reunion(automate1, automate2):
        """Réunion de deux automates"""
        # Unifier les alphabets
        alphabet1 = set(s.strip() for s in automate1.alphabet.split(','))
        alphabet2 = set(s.strip() for s in automate2.alphabet.split(','))
        alphabet_union = alphabet1.union(alphabet2)
        alphabet_str = ','.join(sorted(alphabet_union))
        
        # Créer un nouvel état initial
        etats_data = [{'nom': 'q_init', 'est_final': False}]
        
        # Ajouter les états des deux automates avec préfixes
        for etat in automate1.etats.all():
            etats_data.append({
                'nom': f"1_{etat.nom}",
                'est_final': etat.est_final
            })
        
        for etat in automate2.etats.all():
            etats_data.append({
                'nom': f"2_{etat.nom}",
                'est_final': etat.est_final
            })
        
        # Créer les transitions
        transitions_data = []
        
        # Epsilon-transitions du nouvel état initial vers les états initiaux
        transitions_data.append({
            'source': 'q_init',
            'destination': f"1_{automate1.etat_initial.nom}",
            'symbole': None  # epsilon
        })
        
        transitions_data.append({
            'source': 'q_init',
            'destination': f"2_{automate2.etat_initial.nom}",
            'symbole': None  # epsilon
        })
        
        # Transitions de l'automate 1
        for trans in automate1.transitions.all():
            transitions_data.append({
                'source': f"1_{trans.source.nom}",
                'destination': f"1_{trans.destination.nom}",
                'symbole': trans.symbole
            })
        
        # Transitions de l'automate 2
        for trans in automate2.transitions.all():
            transitions_data.append({
                'source': f"2_{trans.source.nom}",
                'destination': f"2_{trans.destination.nom}",
                'symbole': trans.symbole
            })
        
        return create_automate(
            nom=f"reunion_{automate1.nom}_{automate2.nom}",
            type='eAFN',
            alphabet=alphabet_str,
            etats_data=etats_data,
            transitions_data=transitions_data,
            etat_initial_id='q_init'
        )
    
    @staticmethod
    def intersection(automate1, automate2):
        """Intersection de deux automates par produit cartésien"""
        # Déterminiser les automates si nécessaire
        if automate1.type != 'AFD':
            automate1 = AutomateOperations.determiniser(automate1)
        if automate2.type != 'AFD':
            automate2 = AutomateOperations.determiniser(automate2)
        
        # Compléter les automates
        automate1 = AutomateOperations.completer(automate1)
        automate2 = AutomateOperations.completer(automate2)
        
        # Unifier les alphabets
        alphabet1 = set(s.strip() for s in automate1.alphabet.split(','))
        alphabet2 = set(s.strip() for s in automate2.alphabet.split(','))
        alphabet_union = alphabet1.union(alphabet2)
        alphabet_str = ','.join(sorted(alphabet_union))
        
        # Construire les tables de transitions
        trans1 = {}
        for etat in automate1.etats.all():
            trans1[etat.nom] = {}
            for symbole in alphabet_union:
                trans1[etat.nom][symbole] = None
        
        for trans in automate1.transitions.all():
            if trans.symbole in alphabet_union:
                trans1[trans.source.nom][trans.symbole] = trans.destination.nom
        
        trans2 = {}
        for etat in automate2.etats.all():
            trans2[etat.nom] = {}
            for symbole in alphabet_union:
                trans2[etat.nom][symbole] = None
        
        for trans in automate2.transitions.all():
            if trans.symbole in alphabet_union:
                trans2[trans.source.nom][trans.symbole] = trans.destination.nom
        
        # Produit cartésien
        etats_data = []
        transitions_data = []
        
        for etat1 in automate1.etats.all():
            for etat2 in automate2.etats.all():
                nom_produit = f"({etat1.nom},{etat2.nom})"
                est_final = etat1.est_final and etat2.est_final
                
                etats_data.append({
                    'nom': nom_produit,
                    'est_final': est_final
                })
                
                # Transitions
                for symbole in alphabet_union:
                    dest1 = trans1[etat1.nom][symbole]
                    dest2 = trans2[etat2.nom][symbole]
                    
                    if dest1 and dest2:
                        transitions_data.append({
                            'source': nom_produit,
                            'destination': f"({dest1},{dest2})",
                            'symbole': symbole
                        })
        
        etat_initial_nom = f"({automate1.etat_initial.nom},{automate2.etat_initial.nom})"
        
        return create_automate(
            nom=f"intersection_{automate1.nom}_{automate2.nom}",
            type='AFD',
            alphabet=alphabet_str,
            etats_data=etats_data,
            transitions_data=transitions_data,
            etat_initial_id=etat_initial_nom
        )
    
    @staticmethod
    def concatenation(automate1, automate2):
        """Concaténation de deux automates"""
        # Unifier les alphabets
        alphabet1 = set(s.strip() for s in automate1.alphabet.split(','))
        alphabet2 = set(s.strip() for s in automate2.alphabet.split(','))
        alphabet_union = alphabet1.union(alphabet2)
        alphabet_str = ','.join(sorted(alphabet_union))
        
        # États du premier automate
        etats_data = []
        for etat in automate1.etats.all():
            etats_data.append({
                'nom': f"1_{etat.nom}",
                'est_final': False  # Les états finaux du premier ne sont plus finaux
            })
        
        # États du second automate
        for etat in automate2.etats.all():
            etats_data.append({
                'nom': f"2_{etat.nom}",
                'est_final': etat.est_final
            })
        
        # Transitions
        transitions_data = []
        
        # Transitions du premier automate
        for trans in automate1.transitions.all():
            transitions_data.append({
                'source': f"1_{trans.source.nom}",
                'destination': f"1_{trans.destination.nom}",
                'symbole': trans.symbole
            })
        
        # Transitions du second automate
        for trans in automate2.transitions.all():
            transitions_data.append({
                'source': f"2_{trans.source.nom}",
                'destination': f"2_{trans.destination.nom}",
                'symbole': trans.symbole
            })
        
        # Epsilon-transitions des états finaux du premier vers l'état initial du second
        for etat in automate1.etats.all():
            if etat.est_final:
                transitions_data.append({
                    'source': f"1_{etat.nom}",
                    'destination': f"2_{automate2.etat_initial.nom}",
                    'symbole': None  # epsilon
                })
        
        return create_automate(
            nom=f"concat_{automate1.nom}_{automate2.nom}",
            type='eAFN',
            alphabet=alphabet_str,
            etats_data=etats_data,
            transitions_data=transitions_data,
            etat_initial_id=f"1_{automate1.etat_initial.nom}"
        )
    
    @staticmethod
    def epsilon_fermeture(automate, etat_nom):
        """Calcul de l'epsilon-fermeture pour un état"""
        closure = set([etat_nom])
        pile = [etat_nom]
        
        # Construire un dictionnaire des epsilon-transitions
        epsilon_trans = defaultdict(list)
        for trans in automate.transitions.all():
            if trans.symbole is None or trans.symbole == '':
                epsilon_trans[trans.source.nom].append(trans.destination.nom)
        
        while pile:
            etat_courant = pile.pop()
            for dest in epsilon_trans[etat_courant]:
                if dest not in closure:
                    closure.add(dest)
                    pile.append(dest)
        
        return list(closure)
    
    @staticmethod
    def etat_accessible(automate):
        """Retourne les états accessibles depuis l'état initial"""
        if not automate.etat_initial:
            return []
        
        accessibles = set()
        pile = [automate.etat_initial.nom]
        
        # Construire un dictionnaire des transitions
        trans_dict = defaultdict(list)
        for trans in automate.transitions.all():
            trans_dict[trans.source.nom].append(trans.destination.nom)
        
        while pile:
            etat_courant = pile.pop()
            if etat_courant not in accessibles:
                accessibles.add(etat_courant)
                for dest in trans_dict[etat_courant]:
                    if dest not in accessibles:
                        pile.append(dest)
        
        return [etat for etat in automate.etats.all() if etat.nom in accessibles]
    
    @staticmethod
    def etat_coaccessible(automate):
        """Retourne les états co-accessibles (qui peuvent atteindre un état final)"""
        # Construire le graphe inverse
        trans_inverse = defaultdict(list)
        for trans in automate.transitions.all():
            trans_inverse[trans.destination.nom].append(trans.source.nom)
        
        # Recherche en arrière depuis les états finaux
        coaccessibles = set()
        pile = []
        
        for etat in automate.etats.all():
            if etat.est_final:
                pile.append(etat.nom)
                coaccessibles.add(etat.nom)
        
        while pile:
            etat_courant = pile.pop()
            for source in trans_inverse[etat_courant]:
                if source not in coaccessibles:
                    coaccessibles.add(source)
                    pile.append(source)
        
        return [etat for etat in automate.etats.all() if etat.nom in coaccessibles]
    
    @staticmethod
    def etat_utile(automate):
        """Retourne les états utiles (accessibles et co-accessibles)"""
        accessibles = set(e.nom for e in AutomateOperations.etat_accessible(automate))
        coaccessibles = set(e.nom for e in AutomateOperations.etat_coaccessible(automate))
        utiles = accessibles.intersection(coaccessibles)
        
        return [etat for etat in automate.etats.all() if etat.nom in utiles]
    
    @staticmethod
    def langage_reconnu(automate):
        """Retourne une description du langage reconnu par l'automate avec son expression régulière (méthode McCluskey)."""

        nb_etats = automate.etats.count()
        nb_finaux = automate.etats.filter(est_final=True).count()
        alphabet = automate.alphabet

        description = f"Langage reconnu par l'automate {automate.nom}:\n"

        if nb_finaux == 0:
            description += "- Langage vide (aucun mot accepté)\n"
        elif automate.etat_initial and automate.etat_initial.est_final:
            description += "- Accepte le mot vide (ε)\n"

        try:
            regex = AutomateOperations.mccluskey(automate)
            description += f"\nexemple d'expression régulière du langage (méthode de McCluskey) :\n{regex}"
        except Exception as e:
            description += f"\nErreur lors du calcul de l'expression régulière : {e}"

        return description
    @staticmethod
    def mccluskey(automate):
        """Retourne l'expression régulière d’un automate via McNaughton-Yamada (McCluskey)."""

        # Construction des structures
        etats = list(automate.etats.all())
        n = len(etats)
        id_map = {etat.id: i for i, etat in enumerate(etats)}
        R = [[set() for _ in range(n)] for _ in range(n)]

        # Initialisation des transitions
        for trans in automate.transitions.all():
            i = id_map[trans.source_id]
            j = id_map[trans.destination_id]
            R[i][j].add(trans.symbole)

        # État initial et états finaux
        q0 = id_map[automate.etat_initial.id]
        finaux = [id_map[etat.id] for etat in etats if etat.est_final]

        # On convertit les ensembles en expressions (union)
        def union(exprs):
            if not exprs:
                return ''
            elif len(exprs) == 1:
                return next(iter(exprs))
            else:
                return '(' + '+'.join(sorted(exprs)) + ')'

        # Initialiser la matrice R[i][j] comme une expression régulière
        R = [[union(R[i][j]) for j in range(n)] for i in range(n)]

        # McNaughton-Yamada : élimination des états intermédiaires
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    rij = R[i][j]
                    rik = R[i][k]
                    rkk = R[k][k]
                    rkj = R[k][j]

                    if rik and rkj:
                        boucle = f"({rkk})*" if rkk else ""
                        nouveau = f"{rik}{boucle}{rkj}"
                        if rij:
                            R[i][j] = f"({rij}+{nouveau})"
                        else:
                            R[i][j] = nouveau

        # Finaliser : l'expression de q0 vers chaque final
        expressions = [R[q0][qf] for qf in finaux if R[q0][qf]]
        return union(set(expressions)) or '∅'




class ExpressionReguliereOperations:
    @staticmethod
    def vers_automate(expression, methode='thompson'):
    # def vers_automate(expression, methode='thompson'):

        """Construction d'un automate à partir d'une expression régulière"""
        if methode == 'thompson':
            return ExpressionReguliereOperations._thompson(expression)
        elif methode == 'glushkov':
            return ExpressionReguliereOperations._glushkov(expression)
        else:
            raise ValueError("Méthode non supportée")
    
    @staticmethod
    def _thompson(expression):
        import itertools

        class Etat:
            _id_counter = itertools.count()
            def __init__(self, est_final=False):
                self.nom = f"q{next(Etat._id_counter)}"
                self.est_final = est_final

        class Automate:
            def __init__(self, etat_initial, etats, transitions, alphabet):
                self.etat_initial = etat_initial
                self.etats = etats
                self.transitions = transitions
                self.alphabet = alphabet

            def to_dict(self):
                return {
                    "etat_initial_id": self.etat_initial.nom,
                    "etats_data": [{'nom': e.nom, 'est_final': e.est_final} for e in self.etats],
                    "transitions_data": self.transitions,
                    "alphabet": ','.join(sorted(self.alphabet)),
                    "type": "eAFN",
                    "nom": "Thompson_result"
                }

        def insert_concat(expression):
            result = ""
            for i in range(len(expression)):
                c1 = expression[i]
                result += c1
                if i + 1 < len(expression):
                    c2 = expression[i + 1]
                    if (c1.isalpha() or c1 == ')' or c1 == '*') and (c2.isalpha() or c2 == '('):
                        result += '.'
            return result

        def to_postfix(expression):
            precedence = {'*': 3, '.': 2, '+': 1}
            output, stack = [], []
            for c in expression:
                if c.isalpha():
                    output.append(c)
                elif c == '(':
                    stack.append(c)
                elif c == ')':
                    while stack and stack[-1] != '(':
                        output.append(stack.pop())
                    stack.pop()
                else:
                    while stack and stack[-1] != '(' and precedence[c] <= precedence[stack[-1]]:
                        output.append(stack.pop())
                    stack.append(c)
            while stack:
                output.append(stack.pop())
            return output

        def build(postfix):
            stack = []
            for c in postfix:
                if c.isalpha():
                    s1, s2 = Etat(), Etat(est_final=True)
                    stack.append((
                        s1, s2,
                        [s1, s2],
                        [{'source': s1.nom, 'destination': s2.nom, 'symbole': c}],
                        {c}
                    ))
                elif c == '*':
                    i1, f1, etats1, trans1, alpha1 = stack.pop()
                    s, f = Etat(), Etat(est_final=True)
                    f1.est_final = False
                    trans1 += [
                        {'source': s.nom, 'destination': i1.nom, 'symbole': 'ε'},
                        {'source': s.nom, 'destination': f.nom, 'symbole': 'ε'},
                        {'source': f1.nom, 'destination': i1.nom, 'symbole': 'ε'},
                        {'source': f1.nom, 'destination': f.nom, 'symbole': 'ε'}
                    ]
                    stack.append((s, f, etats1 + [s, f], trans1, alpha1))
                elif c == '+':
                    i2, f2, e2, t2, a2 = stack.pop()
                    i1, f1, e1, t1, a1 = stack.pop()
                    s, f = Etat(), Etat(est_final=True)
                    f1.est_final = False
                    f2.est_final = False
                    transitions = t1 + t2 + [
                        {'source': s.nom, 'destination': i1.nom, 'symbole': 'ε'},
                        {'source': s.nom, 'destination': i2.nom, 'symbole': 'ε'},
                        {'source': f1.nom, 'destination': f.nom, 'symbole': 'ε'},
                        {'source': f2.nom, 'destination': f.nom, 'symbole': 'ε'}
                    ]
                    stack.append((s, f, e1 + e2 + [s, f], transitions, a1.union(a2)))
                elif c == '.':
                    i2, f2, e2, t2, a2 = stack.pop()
                    i1, f1, e1, t1, a1 = stack.pop()
                    f1.est_final = False
                    transitions = t1 + t2 + [
                        {'source': f1.nom, 'destination': i2.nom, 'symbole': 'ε'}
                    ]
                    stack.append((i1, f2, e1 + e2, transitions, a1.union(a2)))
            return stack[0]

        infix = insert_concat(expression)
        postfix = to_postfix(infix)
        i, f, etats, transitions, alphabet = build(postfix)
        return create_automate(
            nom=f"Thompson_{expression}",
            type="eAFN",
            alphabet=",".join(sorted(alphabet)),
            etats_data=[{'nom': e.nom, 'est_final': e.est_final} for e in etats],
            transitions_data=transitions,
            etat_initial_id=i.nom
        )

    
    @staticmethod
    def _glushkov(expression):
        import itertools

        class Node:
            """Unité de syntaxe avec position unique pour chaque symbole terminal"""
            _pos_counter = itertools.count(1)
            def __init__(self, val, nullable=False, children=None):
                self.val = val
                self.children = children or []
                self.nullable = nullable
                self.firstpos = set()
                self.lastpos = set()
                self.followpos = {}
                self.pos = None
                if val.isalpha():
                    self.pos = next(Node._pos_counter)

        # Étapes de parsing (ajouter concat explicite, postfix, arbre syntaxique)
        def insert_concat(expr):
            result = ""
            for i in range(len(expr)):
                c1 = expr[i]
                result += c1
                if i + 1 < len(expr):
                    c2 = expr[i + 1]
                    if (c1.isalpha() or c1 == ')' or c1 == '*') and (c2.isalpha() or c2 == '('):
                        result += '.'
            return result

        def to_postfix(expr):
            precedence = {'*': 3, '.': 2, '+': 1}
            output, stack = [], []
            for c in expr:
                if c.isalpha():
                    output.append(c)
                elif c == '(':
                    stack.append(c)
                elif c == ')':
                    while stack and stack[-1] != '(':
                        output.append(stack.pop())
                    stack.pop()
                else:
                    while stack and stack[-1] != '(' and precedence[c] <= precedence[stack[-1]]:
                        output.append(stack.pop())
                    stack.append(c)
            while stack:
                output.append(stack.pop())
            return output

        def build_tree(postfix):
            stack = []
            for c in postfix:
                if c.isalpha():
                    node = Node(c)
                    node.firstpos = {node.pos}
                    node.lastpos = {node.pos}
                    stack.append(node)
                elif c == '*':
                    child = stack.pop()
                    node = Node('*', nullable=True, children=[child])
                    node.firstpos = child.firstpos
                    node.lastpos = child.lastpos
                    for p in child.lastpos:
                        child.followpos.setdefault(p, set()).update(child.firstpos)
                    node.followpos = child.followpos
                    stack.append(node)
                elif c == '+':
                    right = stack.pop()
                    left = stack.pop()
                    node = Node('+', nullable=left.nullable or right.nullable, children=[left, right])
                    node.firstpos = left.firstpos | right.firstpos
                    node.lastpos = left.lastpos | right.lastpos
                    node.followpos = merge_follow(left.followpos, right.followpos)
                    stack.append(node)
                elif c == '.':
                    right = stack.pop()
                    left = stack.pop()
                    node = Node('.', nullable=left.nullable and right.nullable, children=[left, right])
                    node.firstpos = left.firstpos if not left.nullable else left.firstpos | right.firstpos
                    node.lastpos = right.lastpos if not right.nullable else left.lastpos | right.lastpos
                    node.followpos = merge_follow(left.followpos, right.followpos)
                    for p in left.lastpos:
                        node.followpos.setdefault(p, set()).update(right.firstpos)
                    stack.append(node)
            return stack[0]

        def merge_follow(f1, f2):
            merged = dict(f1)
            for k, v in f2.items():
                merged.setdefault(k, set()).update(v)
            return merged

        # === Étapes Glushkov ===
        expr = insert_concat(expression)
        postfix = to_postfix(expr)
        syntax_tree = build_tree(postfix)

        symbol_pos = {}  # pos -> symbole
        for node in ExpressionReguliereOperations.preorder(syntax_tree):
            if node.pos:
                symbol_pos[node.pos] = node.val

        states = [{'nom': f"q{i}", 'est_final': False} for i in range(len(symbol_pos) + 1)]
        etat_initial = 'q0'

        transitions = []
        for p in syntax_tree.firstpos:
            transitions.append({'source': 'q0', 'destination': f"q{p}", 'symbole': symbol_pos[p]})

        for src, dests in syntax_tree.followpos.items():
            for d in dests:
                transitions.append({'source': f"q{src}", 'destination': f"q{d}", 'symbole': symbol_pos[d]})

        final_states = syntax_tree.lastpos
        for f in final_states:
            for state in states:
                if state['nom'] == f"q{f}":
                    state['est_final'] = True
                    break

        alphabet = sorted(set(symbol_pos.values()))

        return create_automate(
            nom=f"Glushkov_{expression}",
            type='AFN',
            alphabet=','.join(alphabet),
            etats_data=states,
            transitions_data=transitions,
            etat_initial_id=etat_initial
        )

    # Parcours préfixe utile pour extraire tous les nœuds avec positions
    def preorder(node):
        nodes = [node]
        for child in node.children:
            nodes.extend(ExpressionReguliereOperations.preorder(child))
        return nodes

    
    @staticmethod
    def evaluer(expression, mot):
        """Tester si un mot est reconnu par l'expression régulière"""
        try:
            # Préparer le pattern Python à partir de l'expression fournie
            regex = expression
            regex = regex.replace('ε', '')              # ε → chaine vide
            regex = regex.replace('∅', '(?!.*)')        # ∅ → pattern impossible

            # Ancrer en début ET fin pour correspondance exacte
            pattern = re.compile(f'^{regex}$')
            return bool(pattern.match(mot))

        except re.error:
            # Si le pattern n'est pas un regex Python valide, basculer sur l'automate
            try:
                automate = ExpressionReguliereOperations.vers_automate(expression)
                return ExpressionReguliereOperations.tester_mot(automate, mot)
            except Exception:
                return False
    
    @staticmethod
    def tester_mot(automate, mot):
        """Teste si un mot est accepté par l'automate"""
        if not automate.etat_initial:
            return False
        
        # Si l'automate n'est pas déterministe, le déterminiser d'abord
        if automate.type != 'AFD':
            automate = AutomateOperations.determiniser(automate)
        
        # Construire la table de transitions
        alphabet = [s.strip() for s in automate.alphabet.split(',') if s.strip()]
        trans_dict = {}
        
        for etat in automate.etats.all():
            trans_dict[etat.nom] = {}
        
        for trans in automate.transitions.all():
            if trans.symbole in alphabet:
                trans_dict[trans.source.nom][trans.symbole] = trans.destination.nom
        
        # Simuler l'exécution
        etat_courant = automate.etat_initial.nom
        
        for symbole in mot:
            if symbole not in alphabet:
                return False
            
            if symbole not in trans_dict[etat_courant]:
                return False
            
            etat_courant = trans_dict[etat_courant][symbole]
        
        # Vérifier si l'état final est acceptant
        try:
            etat_final = automate.etats.get(nom=etat_courant)
            return etat_final.est_final
        except Etat.DoesNotExist:
            return False
    
    @staticmethod
    def generer_mots(automate, longueur_max=5):
        """Génère des mots acceptés par l'automate jusqu'à une longueur maximale"""
        if not automate.etat_initial:
            return []
        
        # Si l'automate n'est pas déterministe, le déterminiser
        if automate.type != 'AFD':
            automate = AutomateOperations.determiniser(automate)
        
        alphabet = [s.strip() for s in automate.alphabet.split(',') if s.strip()]
        mots_acceptes = []
        
        # Construire la table de transitions
        trans_dict = {}
        for etat in automate.etats.all():
            trans_dict[etat.nom] = {}
        
        for trans in automate.transitions.all():
            if trans.symbole in alphabet:
                trans_dict[trans.source.nom][trans.symbole] = trans.destination.nom
        
        # BFS pour générer les mots
        queue = deque([(automate.etat_initial.nom, "")])
        
        while queue:
            etat_courant, mot_courant = queue.popleft()
            
            # Si l'état courant est final, ajouter le mot
            try:
                etat_obj = automate.etats.get(nom=etat_courant)
                if etat_obj.est_final:
                    mots_acceptes.append(mot_courant if mot_courant else "ε")
            except Etat.DoesNotExist:
                continue
            
            # Si on n'a pas atteint la longueur maximale, continuer l'exploration
            if len(mot_courant) < longueur_max:
                for symbole in alphabet:
                    if symbole in trans_dict[etat_courant]:
                        nouvel_etat = trans_dict[etat_courant][symbole]
                        nouveau_mot = mot_courant + symbole
                        queue.append((nouvel_etat, nouveau_mot))
        
        return sorted(list(set(mots_acceptes)))
    
    @staticmethod
    def equivalence_automates(automate1, automate2):
        """Teste l'équivalence de deux automates"""
        # Méthode : L1 = L2 ssi L1 \ L2 = ∅ et L2 \ L1 = ∅
        # Utilise la formule : L1 \ L2 = L1 ∩ complement(L2)
        
        try:
            # Calculer L1 \ L2
            comp_automate2 = AutomateOperations.complement(automate2)
            diff1 = AutomateOperations.intersection(automate1, comp_automate2)
            
            # Calculer L2 \ L1  
            comp_automate1 = AutomateOperations.complement(automate1)
            diff2 = AutomateOperations.intersection(automate2, comp_automate1)
            
            # Vérifier si les langages de différence sont vides
            vide1 = ExpressionReguliereOperations._est_langage_vide(diff1)
            vide2 = ExpressionReguliereOperations._est_langage_vide(diff2)
            
            return vide1 and vide2
        except:
            return False
    
    @staticmethod
    def _est_langage_vide(automate):
        """Vérifie si le langage de l'automate est vide"""
        # Un langage est vide ssi aucun état final n'est accessible
        etats_accessibles = AutomateOperations.etat_accessible(automate)
        
        for etat in etats_accessibles:
            if etat.est_final:
                return False
        
        return True
    
    @staticmethod
    def vers_expression_reguliere(automate):
        """Convertit un automate en expression régulière (méthode d'élimination d'états)"""
        # Simplification : retourne une description du langage
        if not automate.etat_initial:
            return "∅"
        
        # Vérifier si le langage est vide
        if ExpressionReguliereOperations._est_langage_vide(automate):
            return "∅"
        
        # Cas simple : automate avec un seul état final
        etats_finaux = list(automate.etats.filter(est_final=True))
        alphabet = [s.strip() for s in automate.alphabet.split(',') if s.strip()]
        
        if len(etats_finaux) == 1 and automate.etat_initial.est_final:
            # L'automate accepte au moins le mot vide
            if automate.etats.count() == 1:
                # Un seul état qui est initial et final
                transitions_self = automate.transitions.filter(
                    source=automate.etat_initial,
                    destination=automate.etat_initial
                )
                
                if transitions_self.exists():
                    symboles = [t.symbole for t in transitions_self if t.symbole]
                    if symboles:
                        return f"({'+'.join(symboles)})*"
                
                return "ε"
        
        # Pour les cas complexes, retourner une approximation
        transitions = list(automate.transitions.all())
        if len(transitions) == 1 and len(etats_finaux) == 1:
            trans = transitions[0]
            if trans.source == automate.etat_initial and trans.destination.est_final:
                return trans.symbole
        
        # Cas général : construction approximative
        symbols_used = set()
        for trans in transitions:
            if trans.symbole:
                symbols_used.add(trans.symbole)
        
        if symbols_used:
            return f"({'+'.join(sorted(symbols_used))})*"
        else:
            return "ε"


# Fonctions utilitaires supplémentaires

def get_automate(id):
    """Récupère un automate par son ID"""
    try:
        return Automate.objects.get(id=id)
    except Automate.DoesNotExist:
        return None

def update_automate(id, **kwargs):
    """Met à jour un automate"""
    automate = get_automate(id)
    if automate:
        for key, value in kwargs.items():
            setattr(automate, key, value)
        automate.save()
        return automate
    return None

def delete_automate(id):
    """Supprime un automate"""
    automate = get_automate(id)
    if automate:
        automate.delete()
        return True
    return False

def create_expression_reguliere(expression, automate=None, methode_construction=None):
    """Crée une expression régulière"""
    expr = ExpressionReguliere.objects.create(
        expression=expression,
        automate=automate,
        methode_construction=methode_construction
    )
    return expr

def get_expression_reguliere(id):
    """Récupère une expression régulière par son ID"""
    try:
        return ExpressionReguliere.objects.get(id=id)
    except ExpressionReguliere.DoesNotExist:
        return None

def update_expression_reguliere(id, **kwargs):
    """Met à jour une expression régulière"""
    expr = get_expression_reguliere(id)
    if expr:
        for key, value in kwargs.items():
            setattr(expr, key, value)
        expr.save()
        return expr
    return None

def delete_expression_reguliere(id):
    """Supprime une expression régulière"""
    expr = get_expression_reguliere(id)
    if expr:
        expr.delete()
        return True
    return False


# Fonctions de validation et d'analyse

def valider_automate(automate):
    """Valide la cohérence d'un automate"""
    erreurs = []
    
    # Vérifier qu'il y a un état initial
    if not automate.etat_initial:
        erreurs.append("Aucun état initial défini")
    
    # Vérifier que l'état initial appartient à l'automate
    if automate.etat_initial and automate.etat_initial.automate != automate:
        erreurs.append("L'état initial n'appartient pas à cet automate")
    
    # Vérifier qu'il y a au moins un état
    if automate.etats.count() == 0:
        erreurs.append("L'automate n'a aucun état")
    
    # Vérifier que les transitions sont cohérentes
    alphabet = set(s.strip() for s in automate.alphabet.split(',') if s.strip())
    
    for trans in automate.transitions.all():
        if trans.automate != automate:
            erreurs.append(f"Transition incohérente: {trans}")
        
        if trans.source.automate != automate:
            erreurs.append(f"État source incohérent: {trans.source}")
        
        if trans.destination.automate != automate:
            erreurs.append(f"État destination incohérent: {trans.destination}")
        
        if trans.symbole and trans.symbole not in alphabet:
            if automate.type != 'eAFN' or trans.symbole != '':
                erreurs.append(f"Symbole '{trans.symbole}' non dans l'alphabet")
    
    # Vérifications spécifiques au type d'automate
    if automate.type == 'AFD':
        # Vérifier le déterminisme
        trans_dict = defaultdict(lambda: defaultdict(list))
        for trans in automate.transitions.all():
            if trans.symbole:
                trans_dict[trans.source.nom][trans.symbole].append(trans.destination.nom)
        
        for etat_nom, transitions in trans_dict.items():
            for symbole, destinations in transitions.items():
                if len(destinations) > 1:
                    erreurs.append(f"État {etat_nom} non déterministe pour '{symbole}'")
    
    return erreurs

def analyser_automate(automate):
    """Analyse complète d'un automate"""
    analyse = {
        'nom': automate.nom,
        'type': automate.get_type_display(),
        'alphabet': automate.alphabet,
        'nb_etats': automate.etats.count(),
        'nb_transitions': automate.transitions.count(),
        'nb_etats_finaux': automate.etats.filter(est_final=True).count(),
        'est_valide': len(valider_automate(automate)) == 0,
        'erreurs': valider_automate(automate)
    }
    
    try:
        # Analyse des états
        etats_accessibles = AutomateOperations.etat_accessible(automate)
        etats_coaccessibles = AutomateOperations.etat_coaccessible(automate)
        etats_utiles = AutomateOperations.etat_utile(automate)
        
        analyse.update({
            'nb_etats_accessibles': len(etats_accessibles),
            'nb_etats_coaccessibles': len(etats_coaccessibles),
            'nb_etats_utiles': len(etats_utiles),
            'a_etats_inutiles': len(etats_utiles) < automate.etats.count(),
            'langage_vide': ExpressionReguliereOperations._est_langage_vide(automate),
            'accepte_mot_vide': automate.etat_initial and automate.etat_initial.est_final
        })
        
        # Générer quelques mots du langage
        analyse['mots_exemples'] = ExpressionReguliereOperations.generer_mots(automate, 3)
        
    except Exception as e:
        analyse['erreur_analyse'] = str(e)
    
    return analyse

import re
from copy import deepcopy

def clean_expr(expr):
    """Nettoie une expression régulière"""
    if not expr or expr == '':
        return 'ε'
    
    # Supprime les espaces
    expr = expr.replace(' ', '')
    
    # Nettoie les + multiples et en début/fin
    expr = re.sub(r'\++', '+', expr)
    expr = expr.strip('+')
    
    # Gère ε
    if expr == 'ε' or expr == '':
        return 'ε'
    
    return expr

def parse_terms(expr):
    """Sépare une expression en termes"""
    if not expr or expr == 'ε':
        return ['ε']
    
    terms = []
    for term in expr.split('+'):
        term = term.strip()
        if term and term != '':
            terms.append(term)
    
    return terms if terms else ['ε']

def find_var_coefficient(expr, var):
    """Trouve le coefficient de var dans expr"""
    if not expr or expr == 'ε':
        return '', 'ε'
    
    terms = parse_terms(expr)
    coefficients = []
    independent = []
    
    # Pattern pour détecter la variable
    var_pattern = rf'{re.escape(var)}(?![0-9])'  # Évite X1 dans X10
    
    for term in terms:
        if re.search(var_pattern, term):
            # Extrait le coefficient (ce qui précède la variable)
            coeff = re.sub(var_pattern, '', term)
            if coeff == '':
                coeff = 'ε'  # Coefficient unitaire
            coefficients.append(coeff)
        else:
            independent.append(term)
    
    coeff_str = '+'.join(coefficients) if coefficients else ''
    indep_str = '+'.join(independent) if independent else 'ε'
    
    return clean_expr(coeff_str), clean_expr(indep_str)

def apply_arden(var, expr):
    """Applique le lemme d'Arden : X = aX + b → X = a*b"""
    coeff, indep = find_var_coefficient(expr, var)
    
    print(f"  Arden pour {var}: coeff='{coeff}', indep='{indep}'")
    
    if not coeff or coeff == 'ε':
        # Pas de récursion, retourne l'expression nettoyée
        result = clean_expr(expr)
        print(f"  → Pas de récursion: {result}")
        return result
    
    # X = aX + b → X = a*b
    if indep == 'ε':
        if coeff == 'ε':
            result = 'ε'
        else:
            result = f"({coeff})*"
    else:
        if coeff == 'ε':
            result = indep
        else:
            result = f"({coeff})*({indep})"
    
    print(f"  → Lemme d'Arden appliqué: {result}")
    return result

def substitute_var(expr, var, replacement):
    """Substitue une variable par son remplacement dans une expression"""
    if not expr or expr == 'ε':
        return 'ε'
    
    # Pattern pour éviter les substitutions partielles (X1 dans X10)
    var_pattern = rf'{re.escape(var)}(?![0-9])'
    
    # Ajoute des parenthèses au remplacement si nécessaire
    if '+' in replacement and replacement != 'ε':
        replacement = f"({replacement})"
    
    result = re.sub(var_pattern, replacement, expr)
    return clean_expr(result)

def get_variables(expr):
    """Retourne l'ensemble des variables Xi dans l'expression"""
    if not expr:
        return set()
    return set(re.findall(r'X\d+', expr))

def solve_regex_system(equations):
    """Résout complètement le système d'équations régulières"""
    print("=== DÉBUT DE LA RÉSOLUTION ===")
    
    # Copie du système
    system = deepcopy(equations)
    solutions = {}
    
    print("Système initial:")
    for var, expr in system.items():
        print(f"  {var} = {expr}")
    
    # Étape 1: Résolution des équations auto-récursives (X = aX + b)
    print("\n--- Étape 1: Résolution des auto-récursions ---")
    for var in list(system.keys()):
        expr = system[var]
        if var in get_variables(expr):
            print(f"Résolution de {var} = {expr}")
            solution = apply_arden(var, expr)
            solutions[var] = solution
            system[var] = solution
            print(f"  Solution: {var} = {solution}")
    
    # Étape 2: Substitution progressive
    print("\n--- Étape 2: Substitutions progressives ---")
    max_iterations = 50
    
    for iteration in range(max_iterations):
        print(f"\nItération {iteration + 1}:")
        changed = False
        
        # Pour chaque variable
        for var in system.keys():
            old_expr = system[var]
            new_expr = old_expr
            
            # Substitue toutes les autres variables dont on connaît la solution
            for other_var, other_sol in solutions.items():
                if other_var != var and other_var in get_variables(new_expr):
                    print(f"  Substitution de {other_var} dans {var}")
                    print(f"    Avant: {new_expr}")
                    new_expr = substitute_var(new_expr, other_var, other_sol)
                    print(f"    Après: {new_expr}")
            
            # Si l'expression a changé, applique Arden si nécessaire
            if new_expr != old_expr:
                changed = True
                if var in get_variables(new_expr):
                    print(f"  Application d'Arden pour {var}")
                    new_expr = apply_arden(var, new_expr)
                
                system[var] = new_expr
                solutions[var] = new_expr
                print(f"  Mise à jour: {var} = {new_expr}")
        
        # Vérifie si toutes les solutions sont sans variables
        all_resolved = True
        for var, sol in solutions.items():
            if get_variables(sol):
                all_resolved = False
                break
        
        if all_resolved:
            print("  → Toutes les variables sont résolues !")
            break
        
        if not changed:
            print("  → Pas de changement, arrêt des itérations")
            break
    
    # Étape 3: Nettoyage final
    print("\n--- Étape 3: Nettoyage final ---")
    for var in solutions:
        old_sol = solutions[var]
        # Dernière passe de substitution
        for other_var, other_sol in solutions.items():
            if other_var != var:
                solutions[var] = substitute_var(solutions[var], other_var, other_sol)
        
        # Si encore des variables, applique Arden une dernière fois
        if get_variables(solutions[var]):
            solutions[var] = apply_arden(var, solutions[var])
        
        if solutions[var] != old_sol:
            print(f"  Nettoyage final de {var}: {old_sol} → {solutions[var]}")
    
    return solutions

def eliminate_and_simplify(equations):
    """Interface principale"""
    result = solve_regex_system(equations)
    
    print("\n=== SOLUTIONS FINALES ===")
    for var, sol in result.items():
        has_vars = bool(get_variables(sol))
        status = "❌ CONTIENT DES VARIABLES" if has_vars else "✅ RÉSOLU"
        print(f"{var} = {sol} {status}")
    
    return result


