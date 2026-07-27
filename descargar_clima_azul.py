# -*- coding: utf-8 -*-
"""Compatibilidad con el antiguo nombre del actualizador meteorológico.

La lógica única se mantiene en ``actualizar_clima.py`` para evitar que dos
scripts generen series distintas o apliquen tratamientos incompatibles.
"""

from actualizar_clima import main


if __name__ == "__main__":
    main()
