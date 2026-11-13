"""Interface translations (English, Portuguese and French)."""

LANGUAGES = {"English": "en", "Português (BR)": "pt", "Français (CA)": "fr"}

TEXTS = {
    "app_title": {
        "en": "Polyauxic Modeling Platform",
        "pt": "Plataforma de Modelagem Poliauxica",
        "fr": "Plateforme de Modélisation Polyauxique",
    },
    "intro_desc": {
        "en": "This application performs advanced non-linear regression for microbial growth "
        "kinetics. It identifies mono- and polyauxic behaviors using robust statistical "
        "methods (Charbonnier loss, ROUT outlier detection) and selects models via Information "
        "Criteria (AIC, AICc, BIC).",
        "pt": "Este aplicativo realiza regressão não-linear avançada para cinética microbiana. "
        "Identifica comportamentos mono e poliauxicos usando métodos estatísticos robustos "
        "(Charbonnier, outliers ROUT) e seleciona modelos via Critérios de Informação (AIC, "
        "AICc, BIC).",
        "fr": "Cette application effectue une régression non linéaire avancée pour la cinétique "
        "microbienne. Elle identifie les comportements mono- et polyauxiques à l'aide de "
        "méthodes robustes (Charbonnier, ROUT) et sélectionne les modèles via Critères "
        "d'Information (AIC, AICc, BIC).",
    },
    "paper_ref": {
        "en": "Reference Paper & Source:",
        "pt": "Artigo de Referência e Fonte:",
        "fr": "Article de Référence et Source :",
    },
    "zenodo_cite": {
        "en": "Mockaitis, G. (2025). Polyauxic Modeling Platform (v1.0.0) [Software]. Zenodo.",
        "pt": "Mockaitis, G. (2025). Polyauxic Modeling Platform (v1.0.0) [Software]. Zenodo.",
        "fr": "Mockaitis, G. (2025). Polyauxic Modeling Platform (v1.0.0) [Logiciel]. Zenodo.",
    },
    "instructions_header": {
        "en": "Instructions & File Format",
        "pt": "Instruções e Formato do Arquivo",
        "fr": "Instructions et Format de Fichier",
    },
    "instructions_list": {
        "en": "\n"
        "        **Data Preparation & Format:**\n"
        "        * **File Type:** Upload a `.csv` or `.xlsx` (Excel) file.\n"
        "        * **Column Structure (Crucial):** Organize your data in **pairs** of "
        "columns: Time followed by Response.\n"
        "        * **Headers:** The first row must contain the column names.\n"
        "        * **Replicates:** You can include up to **5 biological replicates**. "
        "The system automatically detects them based on the column pairs.\n"
        "        * **Decimals:** Both dot (`.`) and comma (`,`) are accepted.\n"
        "        * **Plot Settings:** You can customize plot font and axis titles in "
        "the sidebar.\n"
        "\n"
        "        **Example Layout:**\n"
        "        | A (Time 1) | B (Resp 1) | C (Time 2) | D (Resp 2) |\n"
        "        | :--- | :--- | :--- | :--- |\n"
        "        | 0.0 | 0.105 | 0.0 | 0.102 |\n"
        "        | 1.0 | 0.200 | 1.0 | 0.198 |\n"
        "        ",
        "pt": "\n"
        "        **Preparação e Formato dos Dados:**\n"
        "        * **Tipo de Arquivo:** Carregue um arquivo `.csv` ou `.xlsx` (Excel).\n"
        "        * **Estrutura das Colunas (Importante):** Organize seus dados "
        "estritamente em **pares**: Tempo seguido de Resposta.\n"
        "        * **Cabeçalho:** A primeira linha deve conter o nome das variáveis.\n"
        "        * **Réplicas:** O sistema aceita até **5 réplicas biológicas**. Basta "
        "adicionar os pares de colunas lado a lado; o sistema os agrupará "
        "automaticamente.\n"
        "        * **Decimais:** Tanto ponto (`.`) quanto vírgula (`,`) são aceitos.\n"
        "        * **Gráficos:** É possível definir a fonte e títulos dos eixos na "
        "barra lateral.\n"
        "\n"
        "        **Exemplo de Layout:**\n"
        "        | A (Tempo 1) | B (Resp 1) | C (Tempo 2) | D (Resp 2) |\n"
        "        | :--- | :--- | :--- | :--- |\n"
        "        | 0.0 | 0.105 | 0.0 | 0.102 |\n"
        "        | 1.0 | 0.200 | 1.0 | 0.198 |\n"
        "        ",
        "fr": "\n"
        "        **Préparation et Format des Données :**\n"
        "        * **Type de Fichier :** Téléchargez un fichier `.csv` ou `.xlsx` "
        "(Excel).\n"
        "        * **Structure des Colonnes (Important) :** Organisez vos données en "
        "**paires** : Temps suivi de Réponse.\n"
        "        * **En-têtes :** La première ligne doit contenir les noms des "
        "colonnes.\n"
        "        * **Réplicats :** Vous pouvez inclure jusqu'à **5 réplicats "
        "biologiques**. Le système les détecte automatiquement.\n"
        "        * **Décimales :** Les points (`.`) et les virgules (`,`) sont "
        "acceptés.\n"
        "        * **Graphiques :** Vous pouvez personnaliser la police et les titres "
        "des axes dans la barre latérale.\n"
        "\n"
        "        **Exemple de mise en page:**\n"
        "        | A (Temps 1) | B (Rep 1) | C (Temps 2) | D (Rep 2) |\n"
        "        | :--- | :--- | :--- | :--- |\n"
        "        | 0.0 | 0.105 | 0.0 | 0.102 |\n"
        "        | 1.0 | 0.200 | 1.0 | 0.198 |\n"
        "        ",
    },
    "sidebar_config": {"en": "Settings", "pt": "Configurações", "fr": "Paramètres"},
    "upload_label": {
        "en": "Upload CSV/XLSX (Col pairs: t1, y1, t2, y2...)",
        "pt": "Arquivo CSV/XLSX (Pares colunas: t1, y1, t2, y2...)",
        "fr": "Télécharger CSV/XLSX (Paires col: t1, y1, t2, y2...)",
    },
    "max_phases": {
        "en": "Max Phases to Test",
        "pt": "Máximo de Fases para testar",
        "fr": "Phases Max à Tester",
    },
    "info_upload": {
        "en": "Please upload a file to start.",
        "pt": "Por favor, carregue um arquivo para começar.",
        "fr": "Veuillez télécharger un fichier para commencer.",
    },
    "data_loaded": {
        "en": "Data Loaded: {0} replicates identified. Total points: {1}",
        "pt": "Dados Carregados: {0} réplicas identificadas. Total de pontos: {1}",
        "fr": "Données Chargées: {0} réplicats identifiés. Points totaux: {1}",
    },
    "run_button": {"en": "RUN ANALYSIS", "pt": "EXECUTAR ANÁLISE", "fr": "LANCER L'ANALYSE"},
    "tab_gompertz": {
        "en": "Gompertz (Eq. 32)",
        "pt": "Gompertz (Eq. 32)",
        "fr": "Gompertz (Eq. 32)",
    },
    "tab_boltzmann": {
        "en": "Boltzmann (Eq. 31)",
        "pt": "Boltzmann (Eq. 31)",
        "fr": "Boltzmann (Eq. 31)",
    },
    "expanding": {
        "en": "{0}: Fitting {1} Phase(s)",
        "pt": "{0}: Ajuste com {1} Fase(s)",
        "fr": "{0}: Ajustement avec {1} Phase(s)",
    },
    "optimizing": {
        "en": "Optimizing {0} phases...",
        "pt": "Otimizando {0} fases...",
        "fr": "Optimisation de {0} phases...",
    },
    "warning_insufficient": {
        "en": "Insufficient data.",
        "pt": "Dados insuficientes.",
        "fr": "Données insuffisantes.",
    },
    "table_title": {
        "en": "Model Selection Table",
        "pt": "Tabela de Seleção de Modelo",
        "fr": "Tableau de Sélection du Modèle",
    },
    "best_model_msg": {
        "en": "🏆 Best Suggested Model: **{0} Phase(s)** (First local minimum of {1}).",
        "pt": "🏆 Melhor Modelo Sugerido: **{0} Fase(s)** (Primeiro mínimo local de {1}).",
        "fr": "🏆 Meilleur Modèle Suggéré : **{0} Phase(s)** (Premier minimum local de {1}).",
    },
    "graph_summary_title": {
        "en": "Effect of Phase Count on Criteria",
        "pt": "Efeito do Número de Fases nos Critérios",
        "fr": "Effet du Nombre de Phases sur les Critères",
    },
    "download_plot": {
        "en": "Download Plot (SVG)",
        "pt": "Baixar Gráfico (SVG)",
        "fr": "Télécharger le Graphique (SVG)",
    },
    "download_summary": {
        "en": "Download Summary (SVG)",
        "pt": "Baixar Resumo (SVG)",
        "fr": "Télécharger le Résumé (SVG)",
    },
    "download_excel": {
        "en": "Download Full Results (.xlsx)",
        "pt": "Baixar Resultados Completos (.xlsx)",
        "fr": "Télécharger les Résultats Complets (.xlsx)",
    },
    "axis_time": {"en": "Time (h/d)", "pt": "Tempo (h/d)", "fr": "Temps (h/j)"},
    "legend_global": {"en": "Global Fit", "pt": "Ajuste Global", "fr": "Ajustement Global"},
    "legend_phase": {"en": "Phase {0}", "pt": "Fase {0}", "fr": "Phase {0}"},
    "legend_mean": {
        "en": "Mean (w/o Outliers)",
        "pt": "Média (s/ Outliers)",
        "fr": "Moyenne (sans Aberrants)",
    },
    "legend_outlier": {"en": "Outliers", "pt": "Outliers", "fr": "Valeurs Aberrantes"},
    "error_read": {
        "en": "Error processing data: {0}",
        "pt": "Erro ao processar dados: {0}",
        "fr": "Erreur de traitement: {0}",
    },
    "error_cols": {"en": "Column error.", "pt": "Erro nas colunas.", "fr": "Erreur de colonne."},
    "error_proc": {
        "en": "Error processing data: {0}",
        "pt": "Erro ao processar dados: {0}",
        "fr": "Erreur de traitement: {0}",
    },
    "sidebar_outlier_header": {
        "en": "Outliers & Robustness",
        "pt": "Outliers e Robustez",
        "fr": "Valeurs Aberrantes & Robustesse",
    },
    "outlier_method_label": {
        "en": "Outlier Removal Method",
        "pt": "Método de Remoção de Outliers",
        "fr": "Méthode de Suppression des Valeurs Aberrantes",
    },
    "outlier_none": {
        "en": "No removal (use all points)",
        "pt": "Nenhuma remoção (usar todos os pontos)",
        "fr": "Aucune suppression (utiliser tous les points)",
    },
    "outlier_simple": {
        "en": "ROUT-like (Simple MAD, Z > 2.5)",
        "pt": "ROUT-like (MAD simples, Z > 2,5)",
        "fr": "ROUT-like (MAD simple, Z > 2,5)",
    },
    "outlier_rout": {
        "en": "ROUT (Robust + FDR)",
        "pt": "ROUT (Robusto + FDR)",
        "fr": "ROUT (Robuste + FDR)",
    },
    "rout_q_label": {
        "en": "ROUT Q (Max FDR %)",
        "pt": "ROUT Q (FDR máx. %)",
        "fr": "ROUT Q (FDR max. %)",
    },
    "constraints_header": {"en": "Constraints", "pt": "Restrições", "fr": "Contraintes"},
    "force_yi": {"en": "Force y_i = 0", "pt": "Forçar y_i = 0", "fr": "Forcer y_i = 0"},
    "force_yf": {"en": "Force y_f = 0", "pt": "Forçar y_f = 0", "fr": "Forcer y_f = 0"},
    "summary_header_used": {"en": "{0} used", "pt": "{0} usado", "fr": "{0} utilisé"},
    "info_selection_criteria": {
        "en": "Model selection criteria: **{0}** (N = {1}, k_min = {2}, k_max = {3}, "
        "N/k_max = {4:.1f}). The selected number of phases is {5} (first local "
        "minimum of {0}).",
        "pt": "Critério de seleção de modelo: **{0}** (N = {1}, k_min = {2}, k_max = "
        "{3}, N/k_max = {4:.1f}). O número de fases selecionado é {5} (primeiro "
        "mínimo local de {0}).",
        "fr": "Critère de sélection du modèle : **{0}** (N = {1}, k_min = {2}, k_max = "
        "{3}, N/k_max = {4:.1f}). Le nombre de phases sélectionné est {5} "
        "(premier minimum local de {0}).",
    },
    "table_col_metric": {"en": "Metric", "pt": "Métrica", "fr": "Métrique"},
    "table_col_value": {"en": "Value", "pt": "Valor", "fr": "Valeur"},
    "table_col_param": {"en": "Param", "pt": "Parâm", "fr": "Param"},
    "table_col_val": {"en": "Val", "pt": "Valor", "fr": "Val"},
    "table_col_se": {"en": "SE", "pt": "EP", "fr": "ET"},
    "table_col_phase": {"en": "Phase", "pt": "Fase", "fr": "Phase"},
    "required_fields_error": {
        "en": "All fields are mandatory.",
        "pt": "Todos os campos são obrigatórios.",
        "fr": "Tous les champs sont obligatoires.",
    },
    "plot_font_label": {
        "en": "Plot Font",
        "pt": "Fonte dos Gráficos",
        "fr": "Police des Graphiques",
    },
    "seed_mode_label": {
        "en": "Random Seed Mode",
        "pt": "Modo da Seed Aleatória",
        "fr": "Mode de Seed Aléatoire",
    },
    "seed_fixed_42": {"en": "Fixed seed (42)", "pt": "Seed fixa (42)", "fr": "Seed fixe (42)"},
    "seed_random": {
        "en": "Random seed (new each run)",
        "pt": "Seed aleatória (nova a cada execução)",
        "fr": "Seed aléatoire (nouvelle à chaque exécution)",
    },
    "seed_used_msg": {
        "en": "Seed used in this run: {0}",
        "pt": "Seed usada nesta execução: {0}",
        "fr": "Seed utilisée dans cette exécution : {0}",
    },
    "axis_x_custom_label": {
        "en": "X-axis title (LaTeX accepted; blank = default)",
        "pt": "Título do eixo X (aceita LaTeX; vazio = padrão)",
        "fr": "Titre de l’axe X (LaTeX accepté; vide = défaut)",
    },
    "axis_y_custom_label": {
        "en": "Y-axis title (LaTeX accepted; blank = default)",
        "pt": "Título do eixo Y (aceita LaTeX; vazio = padrão)",
        "fr": "Titre de l’axe Y (LaTeX accepté; vide = défaut)",
    },
    "axis_math_hint": {
        "en": "Superscript examples: lambda^2, λ^2, \\lambda^2, or $\\lambda^2$.",
        "pt": "Exemplos de sobrescrito: lambda^2, λ^2, \\lambda^2 ou $\\lambda^2$.",
        "fr": "Exemples d’exposant : lambda^2, λ^2, \\lambda^2 ou $\\lambda^2$.",
    },
    "default_y_label": {"en": "Response (y)", "pt": "Resposta (y)", "fr": "Réponse (y)"},
    "metric_corr": {"en": "Correlation (r)", "pt": "Correlação (r)", "fr": "Corrélation (r)"},
    "phase1_col": {"en": "Phase 1", "pt": "Fase 1", "fr": "Phase 1"},
    "phase1_first_order": {"en": "First-order", "pt": "Primeira ordem", "fr": "Premier ordre"},
    "phase1_sigmoidal": {"en": "Sigmoidal", "pt": "Sigmoidal", "fr": "Sigmoïdal"},
    "full_model_sigmoidal": {
        "en": "Sigmoidal ({0})",
        "pt": "Sigmoidal ({0})",
        "fr": "Sigmoïdal ({0})",
    },
    "full_model_first_order": {
        "en": "First-order + {0} (phase 1)",
        "pt": "1ª ordem + {0} (fase 1)",
        "fr": "Premier ordre + {0} (phase 1)",
    },
    "full_model_first_order_only": {"en": "First-order", "pt": "1ª ordem", "fr": "Premier ordre"},
    "full_model_col": {"en": "Full Model", "pt": "Modelo Completo", "fr": "Modèle Complet"},
    "fit_standard_header": {
        "en": "Standard Sigmoidal Fit",
        "pt": "Ajuste Sigmoidal Padrão",
        "fr": "Ajustement Sigmoïdal Standard",
    },
    "fit_first_order_header": {
        "en": "Additional Phase-1 First-Order Fit",
        "pt": "Ajuste Adicional com 1ª Fase de Primeira Ordem",
        "fr": "Ajustement Supplémentaire avec 1ʳᵉ Phase de Premier Ordre",
    },
    "fit_reused_n1": {
        "en": "Reused from {0} single-phase first-order fit (no recomputation).",
        "pt": "Reutilizado do ajuste de fase única em 1ª ordem de {0} (sem novo cálculo).",
        "fr": "Réutilisé depuis l’ajustement mono-phase de premier ordre de {0} (sans recalcul).",
    },
    "fit_overview_header": {
        "en": "Fit Overview",
        "pt": "Visão Geral do Ajuste",
        "fr": "Aperçu de l’Ajustement",
    },
    "details_expander": {
        "en": "Show detailed parameters and diagnostics",
        "pt": "Mostrar parâmetros detalhados e diagnósticos",
        "fr": "Afficher les paramètres détaillés et diagnostics",
    },
    "selection_logic_title": {
        "en": "Automatic Selection Logic",
        "pt": "Lógica Automática de Seleção",
        "fr": "Logique de Sélection Automatique",
    },
    "selection_logic_body": {
        "en": "Criterion in use: **{0}**. For each phase count, the best candidate between "
        "available structures is selected. Then the final phase count follows the "
        "first local minimum rule of the selected criterion.",
        "pt": "Critério em uso: **{0}**. Para cada número de fases, o melhor candidato "
        "entre as estruturas disponíveis é selecionado. Em seguida, o número final "
        "de fases segue a regra do primeiro mínimo local do critério selecionado.",
        "fr": "Critère utilisé : **{0}**. Pour chaque nombre de phases, le meilleur "
        "candidat parmi les structures disponibles est sélectionné. Ensuite, le "
        "nombre final de phases suit la règle du premier minimum local du critère "
        "sélectionné.",
    },
    "legend_phase_best": {
        "en": "Green rows: best candidate within each phase count.",
        "pt": "Linhas verdes: melhor candidato dentro de cada número de fases.",
        "fr": "Lignes vertes : meilleur candidat pour chaque nombre de phases.",
    },
    "legend_final_best": {
        "en": "Blue row: final selected model.",
        "pt": "Linha azul: modelo final selecionado.",
        "fr": "Ligne bleue : modèle final sélectionné.",
    },
    "metrics_marker_sigmoidal": {
        "en": "Sigmoidal phase-1 structure",
        "pt": "Estrutura sigmoidal na fase 1",
        "fr": "Structure sigmoïdale en phase 1",
    },
    "metrics_marker_first_order": {
        "en": "First-order phase-1 structure",
        "pt": "Estrutura de primeira ordem na fase 1",
        "fr": "Structure de premier ordre en phase 1",
    },
    "main_section_adjustments_results": {
        "en": "Adjustments and Results",
        "pt": "Ajustes e Resultados",
        "fr": "Ajustements et Résultats",
    },
    "graphic_options_header": {
        "en": "Graphic Options",
        "pt": "Opções de Gráficos",
        "fr": "Options Graphiques",
    },
    "main_language_label": {"en": "Language", "pt": "Idioma", "fr": "Langue"},
}

# New strings are kept together so all supported languages can be reviewed.
_ADDITIONS = {
    "progress": (
        "Fitting {0}: {1}/{2} phases · {3:.1f} s elapsed. Completed fits appear below; model selection follows at the end.",
        "Ajustando {0}: {1}/{2} fases · {3:.1f} s decorridos. Ajustes concluídos aparecem abaixo; a seleção ocorre ao final.",
        "Ajustement {0} : {1}/{2} phases · {3:.1f} s écoulées. Les ajustements terminés apparaissent ci-dessous; la sélection suit à la fin.",
    ),
    "analysis_duration": (
        "Analysis time: {0:.1f} s (results retained in this session).",
        "Tempo de análise: {0:.1f} s (resultados conservados nesta sessão).",
        "Durée de l’analyse : {0:.1f} s (résultats conservés dans cette session).",
    ),
    "authorship": ("Developed by", "Autoria e desenvolvimento", "Auteur et développement"),
    "automatic": ("Automatic", "Automático", "Automatique"),
    "detected_format": (
        "CSV interpretation: column separator `{0}`; decimal separator `{1}`.",
        "Leitura do CSV: separador de colunas `{0}`; separador decimal `{1}`.",
        "Lecture CSV : séparateur de colonnes `{0}`; séparateur décimal `{1}`.",
    ),
    "mixed_decimal": (
        "Mixed decimal conventions. Use one decimal separator throughout the file, or explicitly select the decimal convention.",
        "Convenções decimais misturadas. Use um único separador decimal no arquivo ou selecione explicitamente a convenção decimal.",
        "Conventions décimales mélangées. Utilisez un seul séparateur dans le fichier ou sélectionnez explicitement la convention décimale.",
    ),
    "parameters_header": (
        "Parameter estimates and standard errors",
        "Parâmetros estimados e erros padrão",
        "Paramètres estimés et erreurs-types",
    ),
    "parameter_help": (
        "Values are shown as estimate ± standard error (6 significant digits). Fixed parameters are not estimated; N/A means the parameter is absent from the model. An unavailable standard error is not zero.",
        "Valores apresentados como estimativa ± erro padrão (6 algarismos significativos). Parâmetros fixos não são estimados; N/A indica parâmetro ausente no modelo. Erro padrão não estimável não é zero.",
        "Valeurs affichées : estimation ± erreur-type (6 chiffres significatifs). Les paramètres fixes ne sont pas estimés; N/A indique un paramètre absent du modèle. Une erreur-type non estimable n’est pas zéro.",
    ),
    "fixed_parameter": ("fixed", "fixo", "fixe"),
    "not_estimable": ("not estimable", "não estimável", "non estimable"),
    "privacy": (
        "Open access. No registration. Uploaded data and results remain only in this session; download results before closing it.",
        "Acesso livre, sem cadastro. Dados enviados e resultados ficam apenas nesta sessão; baixe os resultados antes de fechá-la.",
        "Accès libre, sans inscription. Les données et résultats restent uniquement dans cette session; téléchargez-les avant de la fermer.",
    ),
    "example": (
        "Use synthetic example",
        "Usar exemplo sintético",
        "Utiliser un exemple synthétique",
    ),
    "separator": ("CSV separator", "Separador do CSV", "Séparateur CSV"),
    "decimal": ("Decimal separator", "Separador decimal", "Séparateur décimal"),
    "no_fit": (
        "Insufficient distinct times or no valid fit for this candidate.",
        "Tempos distintos insuficientes ou nenhum ajuste válido para este candidato.",
        "Temps distincts insuffisants ou aucun ajustement valide pour ce candidat.",
    ),
    "no_convergence": (
        "No identifiable, converged candidate. Automatic selection is unavailable.",
        "Nenhum candidato identificável e convergente. Seleção automática indisponível.",
        "Aucun candidat identifiable et convergé. Sélection automatique indisponible.",
    ),
    "different_observations": (
        "Candidates excluded different observations. Their information criteria cannot be ranked automatically. Inspect exclusions or rerun without removal.",
        "Os candidatos excluíram observações diferentes. Seus critérios de informação não podem ser ordenados automaticamente. Examine as exclusões ou repita sem remoção.",
        "Les candidats ont exclu des observations différentes. Leurs critères ne peuvent pas être classés automatiquement. Examinez les exclusions ou relancez sans suppression.",
    ),
    "insufficient_ic": (
        "Insufficient observations for finite information criteria.",
        "Observações insuficientes para critérios de informação finitos.",
        "Observations insuffisantes pour des critères d’information finis.",
    ),
    "timeout": (
        "Computation limit reached. Partial results are exploratory; reduce the number of phases and rerun. No automatic selection was made.",
        "Limite de cálculo atingido. Os resultados parciais são exploratórios; reduza o número de fases e repita. Não houve seleção automática.",
        "Limite de calcul atteinte. Les résultats partiels sont exploratoires; réduisez les phases et relancez. Aucune sélection automatique.",
    ),
    "uncertainty": (
        "Standard errors unavailable: rank deficiency or incomplete convergence prevents a local covariance estimate.",
        "Erros padrão não estimáveis: deficiência de posto ou convergência incompleta impedem estimar a covariância local.",
        "Erreurs-types non estimables : un défaut de rang ou une convergence incomplète empêche une covariance locale.",
    ),
    "uncertainty_boundary": (
        "Active or nearby constraint: {0}. Displayed standard errors are local linear approximations that ignore this constraint. They are not validated confidence intervals; inspect the affected parameters and extrapolation.",
        "Restrição ativa ou próxima: {0}. Os erros padrão exibidos são aproximações lineares locais que desconsideram essa restrição. Não são intervalos de confiança validados; examine os parâmetros afetados e a extrapolação.",
        "Contrainte active ou proche : {0}. Les erreurs-types affichées sont des approximations linéaires locales qui ignorent cette contrainte. Ce ne sont pas des intervalles de confiance validés; examinez les paramètres concernés et l’extrapolation.",
    ),
    "scope": (
        "Increasing responses; time ≥ 0. Use consistent units in every replicate. A zero final asymptote is not a decreasing-kinetics model.",
        "Respostas crescentes; tempo ≥ 0. Use unidades consistentes em todas as réplicas. Fixar a assíntota final em zero não constitui um modelo de cinética decrescente.",
        "Réponses croissantes; temps ≥ 0. Utilisez des unités cohérentes pour toutes les répétitions. Une asymptote finale nulle ne constitue pas un modèle de cinétique décroissante.",
    ),
    "limits": (
        "CSV UTF-8 or XLSX; headers and time/response pairs; ≤5 MB, 5 replicates, 5,000 rows and 10,000 observations. No formulas or thousands separators.",
        "CSV UTF-8 ou XLSX; cabeçalhos e pares tempo/resposta; ≤5 MB, 5 réplicas, 5.000 linhas e 10.000 observações. Sem fórmulas ou separadores de milhares.",
        "CSV UTF-8 ou XLSX; en-têtes et paires temps/réponse; ≤5 Mo, 5 répétitions, 5 000 lignes et 10 000 observations. Sans formules ni séparateurs de milliers.",
    ),
    "file_size": (
        "Empty or oversized file (including expanded XLSX).",
        "Arquivo vazio ou acima do limite, inclusive após descompactar o XLSX.",
        "Fichier vide ou trop volumineux, y compris après décompression XLSX.",
    ),
    "encoding": ("Save the CSV as UTF-8.", "Salve o CSV em UTF-8.", "Enregistrez le CSV en UTF-8."),
    "delimiter": (
        "Select the correct CSV separator.",
        "Selecione o separador correto do CSV.",
        "Sélectionnez le séparateur CSV correct.",
    ),
    "data_size": (
        "Dataset exceeds the row or observation limit.",
        "Dados excedem o limite de linhas ou observações.",
        "Le jeu dépasse la limite de lignes ou observations.",
    ),
    "columns": (
        "Provide 1–5 complete pairs of time/response columns.",
        "Forneça de 1 a 5 pares completos de colunas tempo/resposta.",
        "Fournissez 1 à 5 paires complètes de colonnes temps/réponse.",
    ),
    "workbook": (
        "Encrypted workbooks, macros and external links are not accepted.",
        "Planilhas criptografadas, macros e links externos não são aceitos.",
        "Les classeurs chiffrés, macros et liens externes ne sont pas acceptés.",
    ),
    "formula": (
        "Replace formulas with numeric values before uploading.",
        "Substitua as fórmulas por valores numéricos antes de enviar.",
        "Remplacez les formules par des valeurs numériques avant l’envoi.",
    ),
    "file_type": (
        "Use a CSV or XLSX file.",
        "Utilize um arquivo CSV ou XLSX.",
        "Utilisez un fichier CSV ou XLSX.",
    ),
    "file_parse": (
        "Unable to read this file. Check its format and contents.",
        "Não foi possível ler o arquivo. Confira seu formato e conteúdo.",
        "Impossible de lire le fichier. Vérifiez le format et le contenu.",
    ),
    "empty_data": (
        "No paired observations were found.",
        "Nenhuma observação pareada foi encontrada.",
        "Aucune observation appariée trouvée.",
    ),
    "headers": (
        "The first row must contain column names.",
        "A primeira linha deve conter nomes de colunas.",
        "La première ligne doit contenir des noms de colonnes.",
    ),
    "partial_pair": (
        "A time or response is missing from a pair; fill both or leave both blank.",
        "Falta tempo ou resposta em um par; preencha ambos ou deixe ambos vazios.",
        "Temps ou réponse manquant; remplissez les deux ou laissez les deux vides.",
    ),
    "numeric": (
        "Nonnumeric or infinite value; check the decimal separator.",
        "Valor não numérico ou infinito; confira o separador decimal.",
        "Valeur non numérique ou infinie; vérifiez le séparateur décimal.",
    ),
    "time": (
        "Use nonnegative times and at least two distinct sampling times.",
        "Utilize tempos não negativos e pelo menos dois tempos de coleta distintos.",
        "Utilisez des temps positifs ou nuls et au moins deux temps distincts.",
    ),
    "constant": (
        "Constant responses do not identify kinetic parameters.",
        "Respostas constantes não identificam parâmetros cinéticos.",
        "Des réponses constantes ne permettent pas d’identifier la cinétique.",
    ),
    "unexpected": (
        "Analysis could not be completed. Review the input and reduce the phase limit.",
        "Não foi possível concluir a análise. Confira os dados e reduza o limite de fases.",
        "L’analyse a échoué. Vérifiez les données et réduisez le nombre de phases.",
    ),
    "bad_axis": (
        "Invalid mathematical axis label; the default label is shown.",
        "Rótulo matemático de eixo inválido; exibindo o rótulo padrão.",
        "Libellé mathématique d’axe invalide; utilisation du libellé par défaut.",
    ),
    "raw_title": ("Experimental data", "Dados experimentais", "Données expérimentales"),
    "fit_summary": (
        "Selected fits (selection within each model family)",
        "Ajustes selecionados (seleção dentro de cada família)",
        "Ajustements sélectionnés (sélection dans chaque famille)",
    ),
    "stale": (
        "Settings changed. Run the analysis to obtain results for these conditions.",
        "Configurações alteradas. Execute a análise para obter resultados nestas condições.",
        "Paramètres modifiés. Relancez l’analyse pour ces conditions.",
    ),
}
TEXTS.update(
    {key: dict(zip(("en", "pt", "fr"), values, strict=False)) for key, values in _ADDITIONS.items()}
)

TEXTS["busy"] = {
    "en": "Two analyses are already running. Try again shortly.",
    "pt": "Duas análises já estão em execução. Tente novamente em instantes.",
    "fr": "Deux analyses sont déjà en cours. Réessayez dans quelques instants.",
}
TEXTS["outlier_rout"] = {
    "en": "Robust + FDR (legacy approximation)",
    "pt": "Robusto + FDR (aproximação do legado)",
    "fr": "Robuste + FDR (approximation historique)",
}

TEXTS["candidate"] = {
    "en": "Inspect candidate",
    "pt": "Examinar candidato",
    "fr": "Examiner un candidat",
}
