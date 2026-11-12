"""Public, anonymous Streamlit entry point."""

import sys

if sys.version_info < (3, 12):  # noqa: UP036 - explain migration on the old service runtime
    import streamlit as st

    st.error(
        "This update requires Python 3.12 or later. / Esta atualização exige Python 3.12 ou superior. "
        "Recreate the service virtual environment and install requirements.txt. "
        "See docs/LEGACY_STREAMLIT_MAINTENANCE.md."
    )
    st.stop()
else:
    from legacy_streamlit.ui import main

    if __name__ == "__main__":
        main()
