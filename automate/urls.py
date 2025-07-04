from django.urls import path
from automateApp import views

urlpatterns = [
    path('', views.index, name='index'),

    # Automates
    path('automates/', views.automate_list, name='automate_list'),
    path('automates/<int:id>/', views.automate_detail, name='automate_detail'),
    path('automates/<int:id>/dessiner/', views.dessiner_automate, name='dessiner_automate'),
    path('automates/<int:id>/tester_mot/', views.tester_mot, name='tester_mot'),
    path('automates/<int:id>/langage/', views.langage_reconnu, name='langage_reconnu'),
    path('automates/<int:id>/table_transition/', views.table_transition, name='table_transition'),
    path('automates/<int:id>/<str:operation>/', views.operation_automate, name='operation_automate'),
    path('automates/<int:id1>/<int:id2>/<str:operation>/', views.operation_deux_automates, name='operation_deux_automates'),
    
    # États
    path('automates/<int:id>/etats/<int:etat_id>/epsilon_fermeture/', views.epsilon_fermeture, name='epsilon_fermeture'),
    path('automates/<int:id>/etats/<int:etat_id>/proprietes/', views.proprietes_etat, name='proprietes_etat'),
    
    # Expressions régulières
    path('expressions/construire/', views.construire_expression, name='construire_expression'),
    # systemes d'equations
    path('resoudre_equations_regex/', views.resoudre_equations_regex, name='resoudre_equations'),

]