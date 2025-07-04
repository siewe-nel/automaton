from django.db import models

class Automate(models.Model):
    TYPES = [
        ('AFD', 'Automate fini déterministe'),
        ('AFN', 'Automate fini non déterministe'),
        ('eAFN', 'Automate fini non déterministe avec epsilon-transitions'),
    ]
    nom = models.CharField(max_length=100)
    type = models.CharField(max_length=4, choices=TYPES)
    alphabet = models.CharField(max_length=100)  # Chaîne de caractères séparés par des virgules
    etat_initial = models.ForeignKey('Etat', on_delete=models.CASCADE, related_name='automate_initial', null=True)
    
    def __str__(self):
        return self.nom

class Etat(models.Model):
    automate = models.ForeignKey(Automate, on_delete=models.CASCADE, related_name='etats')
    nom = models.CharField(max_length=50)
    est_final = models.BooleanField(default=False)
    
    def __str__(self):
        return self.nom

class Transition(models.Model):
    automate = models.ForeignKey(Automate, on_delete=models.CASCADE, related_name='transitions')
    source = models.ForeignKey(Etat, on_delete=models.CASCADE, related_name='transitions_source')
    destination = models.ForeignKey(Etat, on_delete=models.CASCADE, related_name='transitions_destination')
    symbole = models.CharField(max_length=1, null=True, blank=True)  # Peut être vide pour epsilon
    
    def __str__(self):
        return f"{self.source} --{self.symbole}--> {self.destination}"

class ExpressionReguliere(models.Model):
    expression = models.CharField(max_length=200)
    automate = models.OneToOneField(Automate, on_delete=models.CASCADE, null=True, blank=True)
    methode_construction = models.CharField(max_length=20, choices=[('thompson', 'Thompson'), ('glushkov', 'Glushkov')], null=True, blank=True)
    
    def __str__(self):
        return self.expression