from django.db import models

class State(models.Model):
    automaton = models.ForeignKey('Automaton', related_name='states', on_delete=models.CASCADE)
    name = models.CharField(max_length=100)
    is_initial = models.BooleanField(default=False)
    is_final = models.BooleanField(default=False)

class Transition(models.Model):
    from_state = models.ForeignKey(State, related_name='out_transitions', on_delete=models.CASCADE)
    to_state = models.ForeignKey(State, related_name='in_transitions', on_delete=models.CASCADE)
    symbol = models.CharField(max_length=10, blank=True)

class Automaton(models.Model):
    type = models.CharField(max_length=10)  # 'AFD', 'AFN', 'e-AFN'
    alphabet = models.CharField(max_length=100)

class RegularExpression(models.Model):
    expression = models.CharField(max_length=200)

class AutomatonClass:
    def __init__(self, states, alphabet, transitions, initial_state, final_states, is_dfa=False):
        self.states = states
        self.alphabet = alphabet
        self.transitions = transitions
        self.initial_state = initial_state
        self.final_states = final_states
        self.is_dfa = is_dfa

def db_to_automaton(automaton_id):
    automaton = Automaton.objects.get(id=automaton_id)
    states = {state.name for state in automaton.states.all()}
    alphabet = set(automaton.alphabet.split(','))
    transitions = {}
    initial_state = None
    final_states = set()
    for state in automaton.states.all():
        if state.is_initial:
            initial_state = state.name
        if state.is_final:
            final_states.add(state.name)
        transitions[state.name] = {}
    for transition in automaton.transitions.all():
        from_state = transition.from_state.name
        symbol = transition.symbol if transition.symbol else ''
        to_state = transition.to_state.name
        if symbol not in transitions[from_state]:
            transitions[from_state][symbol] = set()
        transitions[from_state][symbol].add(to_state)
    is_dfa = automaton.type == 'AFD'
    if is_dfa:
        for state in transitions:
            for symbol in transitions[state]:
                transitions[state][symbol] = next(iter(transitions[state][symbol]))
    return AutomatonClass(states, alphabet, transitions, initial_state, final_states, is_dfa)

def save_automaton(automaton, type):
    alphabet_str = ','.join(automaton.alphabet)
    new_automaton = Automaton.objects.create(type=type, alphabet=alphabet_str)
    state_map = {}
    for state in automaton.states:
        is_initial = state == automaton.initial_state
        is_final = state in automaton.final_states
        db_state = State.objects.create(automaton=new_automaton, name=state, is_initial=is_initial, is_final=is_final)
        state_map[state] = db_state
    if automaton.is_dfa:
        for from_state, trans in automaton.transitions.items():
            for symbol, to_state in trans.items():
                Transition.objects.create(
                    from_state=state_map[from_state],
                    to_state=state_map[to_state],
                    symbol=symbol if symbol else ''
                )
    else:
        for from_state, trans in automaton.transitions.items():
            for symbol, to_states in trans.items():
                for to_state in to_states:
                    Transition.objects.create(
                        from_state=state_map[from_state],
                        to_state=state_map[to_state],
                        symbol=symbol if symbol else ''
                    )
    return new_automaton.id

def epsilon_closure(automaton, states):
    if automaton.is_dfa:
        return states
    closure = set(states)
    stack = list(states)
    while stack:
        state = stack.pop()
        for next_state in automaton.transitions.get(state, {}).get('', set()):
            if next_state not in closure:
                closure.add(next_state)
                stack.append(next_state)
    return closure

def determinize(automaton):
    if automaton.is_dfa:
        return automaton
    initial_closure = epsilon_closure(automaton, {automaton.initial_state})
    new_states = [frozenset(initial_closure)]
    new_transitions = {}
    new_final_states = set()
    stack = [frozenset(initial_closure)]
    state_map = {frozenset(initial_closure): 'q0'}
    counter = 1
    while stack:
        current_set = stack.pop()
        current_state = state_map[current_set]
        if any(state in automaton.final_states for state in current_set):
            new_final_states.add(current_state)
        for symbol in automaton.alphabet:
            next_set = set()
            for state in current_set:
                next_set.update(automaton.transitions.get(state, {}).get(symbol, set()))
            next_closure = epsilon_closure(automaton, next_set)
            if next_closure:
                next_frozenset = frozenset(next_closure)
                if next_frozenset not in state_map:
                    state_map[next_frozenset] = f'q{counter}'
                    counter += 1
                    stack.append(next_frozenset)
                new_transitions.setdefault(current_state, {})[symbol] = state_map[next_frozenset]
    return AutomatonClass(
        states=set(state_map.values()),
        alphabet=automaton.alphabet,
        transitions=new_transitions,
        initial_state='q0',
        final_states=new_final_states,
        is_dfa=True
    )

def minimize(automaton):
    if not automaton.is_dfa:
        automaton = determinize(automaton)
    P = [automaton.final_states, automaton.states - automaton.final_states]
    W = [automaton.final_states.copy()]
    while W:
        A = W.pop()
        for symbol in automaton.alphabet:
            X = set()
            for state in automaton.states:
                next_state = automaton.transitions.get(state, {}).get(symbol)
                if next_state in A:
                    X.add(state)
            for Y in P[:]:
                inter = Y & X
                diff = Y - X
                if inter and diff:
                    P.remove(Y)
                    P.append(inter)
                    P.append(diff)
                    if Y in W:
                        W.remove(Y)
                        W.append(inter)
                        W.append(diff)
                    else:
                        W.append(inter if len(inter) <= len(diff) else diff)
    state_map = {}
    new_states = set()
    new_transitions = {}
    new_final_states = set()
    new_initial_state = None
    for i, group in enumerate(P):
        new_state = f'q{i}'
        new_states.add(new_state)
        if automaton.initial_state in group:
            new_initial_state = new_state
        if group & automaton.final_states:
            new_final_states.add(new_state)
        rep_state = next(iter(group))
        state_map[rep_state] = new_state
        new_transitions[new_state] = {}
        for symbol in automaton.alphabet:
            next_state = automaton.transitions.get(rep_state, {}).get(symbol)
            if next_state:
                for g in P:
                    if next_state in g:
                        new_transitions[new_state][symbol] = state_map[next(iter(g))]
                        break
    return AutomatonClass(new_states, automaton.alphabet, new_transitions, new_initial_state, new_final_states, True)

def thompson_construction(regex):
    states = set()
    transitions = {}
    counter = 0
    def new_state():
        nonlocal counter
        state = f'q{counter}'
        counter += 1
        states.add(state)
        transitions[state] = {}
        return state
    stack = []
    for char in regex:
        if char == '*':
            nfa = stack.pop()
            start = new_state()
            end = new_state()
            transitions[start][''] = {nfa.initial_state, end}
            for s in nfa.final_states:
                transitions[s][''] = {nfa.initial_state, end}
            stack.append(AutomatonClass(states, nfa.alphabet, transitions, start, {end}, False))
        elif char == '|':
            nfa2 = stack.pop()
            nfa1 = stack.pop()
            start = new_state()
            end = new_state()
            transitions[start][''] = {nfa1.initial_state, nfa2.initial_state}
            for s in nfa1.final_states | nfa2.final_states:
                transitions[s][''] = {end}
            states.update(nfa1.states | nfa2.states)
            stack.append(AutomatonClass(states, nfa1.alphabet | nfa2.alphabet, transitions, start, {end}, False))
        elif char == '.':
            nfa2 = stack.pop()
            nfa1 = stack.pop()
            for s in nfa1.final_states:
                transitions[s][''] = {nfa2.initial_state}
            states.update(nfa1.states | nfa2.states)
            stack.append(AutomatonClass(states, nfa1.alphabet | nfa2.alphabet, transitions, nfa1.initial_state, nfa2.final_states, False))
        else:
            start = new_state()
            end = new_state()
            transitions[start][char] = {end}
            stack.append(AutomatonClass(states, {char}, transitions, start, {end}, False))
    return stack[0]

def union(aut1, aut2):
    new_states = {f'1_{s}' for s in aut1.states} | {f'2_{s}' for s in aut2.states}
    new_transitions = {}

    for s, t in aut1.transitions.items():
        new_transitions[f'1_{s}'] = {
            k: {f'1_{v}'} if aut1.is_dfa else {f'1_{x}' for x in v}
            for k, v in t.items()
        }

    for s, t in aut2.transitions.items():
        new_transitions[f'2_{s}'] = {
            k: {f'2_{v}'} if aut2.is_dfa else {f'2_{x}' for x in v}
            for k, v in t.items()
        }

    start = 'start'
    new_states.add(start)
    new_transitions[start] = {'': {f'1_{aut1.initial_state}', f'2_{aut2.initial_state}'}}
    
    new_final_states = {f'1_{s}' for s in aut1.final_states} | {f'2_{s}' for s in aut2.final_states}
    
    return AutomatonClass(
        new_states,
        aut1.alphabet | aut2.alphabet,
        new_transitions,
        start,
        new_final_states,
        False
    )

def intersection(aut1, aut2):
    aut1 = determinize(aut1) if not aut1.is_dfa else aut1
    aut2 = determinize(aut2) if not aut2.is_dfa else aut2
    new_states = {f'{s1},{s2}' for s1 in aut1.states for s2 in aut2.states}
    new_transitions = {}
    for s1 in aut1.states:
        for s2 in aut2.states:
            state = f'{s1},{s2}'
            new_transitions[state] = {}
            for symbol in aut1.alphabet & aut2.alphabet:
                next1 = aut1.transitions.get(s1, {}).get(symbol)
                next2 = aut2.transitions.get(s2, {}).get(symbol)
                if next1 and next2:
                    new_transitions[state][symbol] = f'{next1},{next2}'
    new_initial_state = f'{aut1.initial_state},{aut2.initial_state}'
    new_final_states = {f'{s1},{s2}' for s1 in aut1.final_states for s2 in aut2.final_states}
    return AutomatonClass(new_states, aut1.alphabet & aut2.alphabet, new_transitions, new_initial_state, new_final_states, True)

def complement(aut):
    if not aut.is_dfa:
        aut = determinize(aut)
    return AutomatonClass(aut.states, aut.alphabet, aut.transitions, aut.initial_state, aut.states - aut.final_states, True)

def concatenate(aut1, aut2):
    new_states = {f'1_{s}' for s in aut1.states} | {f'2_{s}' for s in aut2.states}
    new_transitions = {}

    for s, t in aut1.transitions.items():
        new_transitions[f'1_{s}'] = {
            k: {f'1_{v}'} if aut1.is_dfa else {f'1_{x}' for x in v}
            for k, v in t.items()
        }

    for s, t in aut2.transitions.items():
        new_transitions[f'2_{s}'] = {
            k: {f'2_{v}'} if aut2.is_dfa else {f'2_{x}' for x in v}
            for k, v in t.items()
        }

    for s in aut1.final_states:
        key = f'1_{s}'
        if '' in new_transitions.get(key, {}):
            new_transitions[key][''].add(f'2_{aut2.initial_state}')
        else:
            new_transitions.setdefault(key, {})[''] = {f'2_{aut2.initial_state}'}

    return AutomatonClass(
        new_states,
        aut1.alphabet | aut2.alphabet,
        new_transitions,
        f'1_{aut1.initial_state}',
        {f'2_{s}' for s in aut2.final_states},
        False
    )

def complete(aut):
    if not aut.is_dfa:
        aut = determinize(aut)
    sink = 'sink'
    new_states = aut.states | {sink}
    new_transitions = {s: t.copy() for s, t in aut.transitions.items()}
    new_transitions[sink] = {symbol: sink for symbol in aut.alphabet}
    for state in new_states:
        for symbol in aut.alphabet:
            if symbol not in new_transitions.get(state, {}):
                new_transitions[state][symbol] = sink
    return AutomatonClass(new_states, aut.alphabet, new_transitions, aut.initial_state, aut.final_states, True)

def get_transition_table(aut):
    table = {}
    for state in aut.states:
        table[state] = {}
        for symbol in aut.alphabet:
            if aut.is_dfa:
                table[state][symbol] = aut.transitions.get(state, {}).get(symbol, None)
            else:
                table[state][symbol] = aut.transitions.get(state, {}).get(symbol, set())
    return table

def accepts_word(aut, word):
    current_states = epsilon_closure(aut, {aut.initial_state}) if not aut.is_dfa else {aut.initial_state}
    for symbol in word:
        next_states = set()
        for state in current_states:
            if aut.is_dfa:
                next_state = aut.transitions.get(state, {}).get(symbol)
                if next_state:
                    next_states.add(next_state)
            else:
                next_states.update(epsilon_closure(aut, aut.transitions.get(state, {}).get(symbol, set())))
        current_states = next_states
    return bool(current_states & aut.final_states)

def is_accessible(aut, state):
    reachable = {aut.initial_state}
    stack = [aut.initial_state]
    while stack:
        current = stack.pop()
        for symbol in aut.alphabet | {''}:
            if aut.is_dfa:
                next_state = aut.transitions.get(current, {}).get(symbol)
                if next_state and next_state not in reachable:
                    reachable.add(next_state)
                    stack.append(next_state)
            else:
                for next_state in aut.transitions.get(current, {}).get(symbol, set()):
                    if next_state not in reachable:
                        reachable.add(next_state)
                        stack.append(next_state)
    return state in reachable

def is_coaccessible(aut, state):
    reachable = set(aut.final_states)
    stack = list(aut.final_states)
    while stack:
        current = stack.pop()
        for from_state in aut.states:
            if aut.is_dfa:
                for symbol, to_state in aut.transitions.get(from_state, {}).items():
                    if to_state == current and from_state not in reachable:
                        reachable.add(from_state)
                        stack.append(from_state)
            else:
                for symbol, to_states in aut.transitions.get(from_state, {}).items():
                    if current in to_states and from_state not in reachable:
                        reachable.add(from_state)
                        stack.append(from_state)
    return state in reachable

def is_useful(aut, state):
    return is_accessible(aut, state) and is_coaccessible(aut, state)

def convert_to_dfa(automaton_id):
    aut = db_to_automaton(automaton_id)
    dfa = determinize(aut)
    return save_automaton(dfa, 'AFD')

def minimize_automaton(automaton_id):
    aut = db_to_automaton(automaton_id)
    min_aut = minimize(aut)
    return save_automaton(min_aut, 'AFD')

def build_automaton_from_regex(regex_id, method='thompson'):
    regex = RegularExpression.objects.get(id=regex_id)
    aut = thompson_construction(regex.expression) if method == 'thompson' else thompson_construction(regex.expression)  # Glushkov à implémenter séparément
    return save_automaton(aut, 'e-AFN')

def union_automata(automaton_id1, automaton_id2):
    aut1 = db_to_automaton(automaton_id1)
    aut2 = db_to_automaton(automaton_id2)
    union_aut = union(aut1, aut2)
    return save_automaton(union_aut, 'e-AFN')

def intersection_automata(automaton_id1, automaton_id2):
    aut1 = db_to_automaton(automaton_id1)
    aut2 = db_to_automaton(automaton_id2)
    intersect_aut = intersection(aut1, aut2)
    return save_automaton(intersect_aut, 'AFD')

def complement_automaton(automaton_id):
    aut = db_to_automaton(automaton_id)
    complement_aut = complement(aut)
    return save_automaton(complement_aut, 'AFD')

def concatenate_automata(automaton_id1, automaton_id2):
    aut1 = db_to_automaton(automaton_id1)
    aut2 = db_to_automaton(automaton_id2)
    concat_aut = concatenate(aut1, aut2)
    return save_automaton(concat_aut, 'e-AFN')

def complete_automaton(automaton_id):
    aut = db_to_automaton(automaton_id)
    complete_aut = complete(aut)
    return save_automaton(complete_aut, 'AFD')

def get_transition_table_view(automaton_id):
    aut = db_to_automaton(automaton_id)
    return get_transition_table(aut)

def test_word(automaton_id, word):
    aut = db_to_automaton(automaton_id)
    return accepts_word(aut, word)

def is_accessible_state(automaton_id, state_name):
    aut = db_to_automaton(automaton_id)
    return is_accessible(aut, state_name)

def is_coaccessible_state(automaton_id, state_name):
    aut = db_to_automaton(automaton_id)
    return is_coaccessible(aut, state_name)

def is_useful_state(automaton_id, state_name):
    aut = db_to_automaton(automaton_id)
    return is_useful(aut, state_name)