# Rapport du BE

[Consulter ou télécharger le PDF final](BE_ML_Rapport.pdf).

Le PDF fourni par le groupe a été copié **sans modification**. Son empreinte
est enregistrée dans [provenance.json](../outputs/tables/provenance.json).

`main.tex` est la source LaTeX locale disponible au moment du nettoyage. Son
texte est antérieur à certaines reformulations du PDF final : ce n'est pas une
garantie de reconstruction à l'identique du document remis. Les chemins vers
les cinq figures ont été adaptés à `../figures/` ; l'illustration Infologic
dans `assets/` est extraite du PDF fourni.

Pour compiler cette source disponible, depuis `report/`, avec une distribution
LaTeX configurée et ses paquets français :

```bash
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Sans la classe institutionnelle `rapportECL.cls`, la source utilise une mise
en page `article`. Le résultat est `main.pdf`, distinct du PDF final conservé.
