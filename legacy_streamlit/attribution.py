"""Author and scientific references preserved from the original public application."""

import streamlit as st

from .i18n import TEXTS

PROFILE_LINKS = (
    ("ORCID", "https://orcid.org/0000-0002-4231-1056"),
    ("Google Scholar", "https://scholar.google.com/citations?user=yR3UvuoAAAAJ&hl=en&oi=ao"),
    ("ResearchGate", "https://www.researchgate.net/profile/Gustavo-Mockaitis"),
    ("Lattes CV", "http://lattes.cnpq.br/1400402042483439"),
    ("LinkedIn", "https://www.linkedin.com/in/gustavo-mockaitis/"),
    ("Web of Science", "https://www.webofscience.com/wos/author/record/J-7107-2019"),
    ("UNICAMP", "http://feagri.unicamp.br/mockaitis"),
)


def render_attribution(lang):
    st.divider()
    st.subheader(TEXTS["authorship"][lang])
    portrait, biography = st.columns([1, 6], vertical_alignment="center")
    with portrait:
        st.image("https://github.com/gusmock.png", width=90)
    with biography:
        st.markdown("**Prof. Dr. Gustavo Mockaitis**")
        st.markdown(
            "**GBMA / FEAGRI / UNICAMP**  \n"
            "Interdisciplinary Research Group of Biotechnology Applied to the Agriculture "
            "and Environment  \n"
            "School of Agricultural Engineering, University of Campinas.  \n"
            "Campinas, SP, Brazil."
        )
    st.markdown(" · ".join(f"[{name}]({url})" for name, url in PROFILE_LINKS))

    st.subheader(TEXTS["paper_ref"][lang])
    st.markdown(
        "Mockaitis, G. (2026). **Mono- and Polyauxic Growth Kinetics: A Semi-Mechanistic "
        "Framework for Complex Biological Dynamics.** *Bulletin of Mathematical Biology*, "
        "88:55. [DOI: 10.1007/s11538-026-01621-7](https://doi.org/10.1007/s11538-026-01621-7). "
        "[PlumX](https://plu.mx/plum/a/?doi=10.1007/s11538-026-01621-7)"
    )
    st.markdown(
        "Mockaitis, G. (2025). **Mono- and Polyauxic Growth Kinetics: A Semi-Mechanistic "
        "Framework for Complex Biological Dynamics.** arXiv: 2507.05960, 42 p. "
        "[DOI: 10.48550/arXiv.2507.05960](https://doi.org/10.48550/arXiv.2507.05960). "
        "[PlumX](https://plu.mx/plum/a/?arxiv=2507.05960)"
    )
    st.markdown(
        TEXTS["zenodo_cite"][lang]
        + " [DOI: 10.5281/zenodo.18025828](https://doi.org/10.5281/zenodo.18025828). "
        "[Code Ocean: 10.24433/CO.0225069.v1](https://doi.org/10.24433/CO.0225069.v1) · "
        "[GitHub](https://github.com/gusmock/mono_polyauxic_kinetics)"
    )
