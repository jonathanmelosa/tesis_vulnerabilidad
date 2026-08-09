"""
Encuesta Longitudinal Colombiana (ELCA)
Limpieza de la base de personas (olas 2010, 2013 y 2016).

Contexto
--------
Este script parte del output de
src/01_download/01_descarga_ELCA/06_consolidacion_bases_personas.py
(data/processed/personas_elca_longitudinal.parquet), que documentaba la
corrupción de codificación de 2013 como acotada a "texto libre y un par de
etiquetas de opción abierta" ("Otra, ¿cuál?"). Al construir las variables de
composición del hogar para el modelo benchmark de prediccion de pobreza
(ver docs/decisions.md, seccion "Metodologia del modelo benchmark") se
encontró que esa descripción era incompleta.

Problema identificado
-----------------------
El caracter de reemplazo Unicode "�" (U+FFFD) reemplaza vocales acentuadas
(á, é, í, ó, ú) y otros caracteres especiales del español en **535 columnas**
de personas_elca_longitudinal.parquet -- 180 en ola 1 (2010) y 435 en ola 2
(2013); ola 3 (2016) no tiene ningun caso. Esto incluye variables
categoricas analiticamente relevantes que el docstring de 06_... describia
como limpias en 2013: `parentesco`, `etnia`, `afiliacion`, entre otras.

Se verifico contra los bytes crudos de UPersonas-csv.tab 2013 (ver
docs/decisions.md) que la corrupcion YA esta en el archivo fuente que
distribuye la ELCA -- no es un problema de como se lee el archivo (una
codificacion equivocada seria corregible re-leyendo con el encoding
correcto), es una perdida de informacion irreversible a nivel de caracter:
el byte original que representaba la vocal acentuada fue reemplazado por
el caracter de reemplazo UTF-8 (EF BF BD) antes de que este pipeline
tuviera acceso al archivo. La unica forma de recuperar el valor correcto es
por contexto: comparar contra la misma categoria aparecida sin corrupcion
en otra fila/ola, o contra el diccionario de la encuesta.

Alcance de esta limpieza (deliberadamente acotado, ver "Lo que NO se
corrige" mas abajo)
---------------------------------------------------------------------------
De las 535 columnas afectadas, 484 tienen vocabulario cerrado (<=25
categorias distintas) y 51 son de alta cardinalidad (texto libre:
descripciones de oficio, nombres de programas sociales, etc.). Este script:

  1. Corrige automaticamente, dentro de cada columna de vocabulario
     cerrado, los valores corruptos que tienen EXACTAMENTE UN candidato
     limpio compatible en la misma columna (en cualquier ola) --
     construyendo un patron donde cada "�" es un comodin de un caracter y
     buscando que valor limpio de esa misma columna calza exactamente. Es
     automatico y reproducible (no depende de una lista escrita a mano),
     pero deliberadamente conservador: si hay 0 o mas de 1 candidato, NO
     se corrige (se deja documentado, ver mas abajo).
  2. Corrige la ambiguedad mas frecuente por separado: cuando el unico
     comodin resuelve a {"Si", "Sí"} (con o sin tilde), se usa "Sí" -- la
     grafia que usa la ELCA de forma consistente en el resto del
     cuestionario (ver SI_TOKENS en build_ingreso_hogar.py).
  3. Corrige a mano, verificado contra ortografia estandar del español,
     las variables priorizadas para el modelo benchmark que la correccion
     automatica no pudo resolver (ningun candidato limpio en la misma
     columna). Ver CORRECCIONES_MANUALES_PRIORITARIAS -- esta lista crece
     a medida que se van necesitando mas variables del modulo Personas
     (ver seccion "Este es un script vivo" mas abajo); no es una lista
     cerrada de una sola vez.

Resultado: las variables en CORRECCIONES_MANUALES_PRIORITARIAS quedan
100% libres de "�" al terminar este script (se valida con assert por
columna). El resto de columnas de vocabulario cerrado quedan mayormente
corregidas (ver reporte impreso al ejecutar); las columnas de texto libre
NO se tocan en absoluto.

Lo que NO se corrige (documentado, no oculto)
------------------------------------------------
De los 956 valores corruptos en columnas de vocabulario cerrado, 322 se
resuelven automaticamente (puntos 1-2) y ~28 mas con la lista manual
(punto 3). Los ~600 restantes, repartidos en ~270 columnas que HOY no son
candidatas para el benchmark, no tienen ningun valor limpio equivalente en
ningun lado del dataset para inferir la correccion -- resolverlos
requeriria consultar, columna por columna, los diccionarios PDF especificos
de cada ola. Se deja como tarea pendiente, a resolver cuando (si) esas
columnas se necesiten; el reporte completo de que quedo sin resolver se
guarda en docs/variable_audit/personas_corrupcion_residual.csv para que la
proxima vez que se necesite una de esas columnas no haya que re-descubrir
el problema desde cero. Las 51 columnas de texto libre (>25 categorias)
tampoco se tocan -- mismo criterio que el proyecto ya aplica en otros
modulos (ARMONIZACION_ARTICULOS, CORRECCIONES_2013 de
06_consolidacion_bases_personas.py): no vale la pena un diccionario de
miles de palabras para campos que no se usan como variable categorica.

Este es un script "vivo": si en el futuro se necesita una columna hoy
pendiente, agregar su correccion en CORRECCIONES_MANUALES_PRIORITARIAS (o
investigar por que la correccion automatica no la resolvio) en vez de
trabajar alrededor del problema en el script de features que la use.
"""

from pathlib import Path
import re

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "personas_elca_longitudinal.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "personas_elca_longitudinal_clean.parquet"
DOC_PATH = PROJECT_ROOT / "docs" / "variable_audit" / "personas_corrupcion_residual.csv"

CARDINALIDAD_MAXIMA_CERRADA = 25
MARCADOR = "�"  # "�"

# ─── Correcciones manuales: variables prioritarias sin match automatico ──────
# Verificadas contra ortografia estandar del español. `ocupacion`/`parentesco`
# tienen demasiadas categorias (>25) para entrar al barrido automatico por
# vocabulario cerrado, aunque su cardinalidad real es manejable a mano.

CORRECCIONES_MANUALES_PRIORITARIAS = {
    "parentesco": {
        # cardinalidad (26) > CARDINALIDAD_MAXIMA_CERRADA: nunca pasa por la
        # correccion automatica, se listan aqui las 5 corruptas completas.
        "C�nyuge o compa�era(o)": "Cónyuge o compañera(o)",
        "Servicio dom�stico, cuidandero y sus parientes": "Servicio doméstico, cuidandero y sus parientes",
        "T�o(a)": "Tío(a)",
        "Nieto(a)del jefe del hogar o de su c�nyuge": "Nieto(a)del jefe del hogar o de su cónyuge",
        "Bisnieto del jefe del hogar o de su c�nyuge": "Bisnieto del jefe del hogar o de su cónyuge",
    },
    "ocupacion": {
        # cardinalidad (31) > CARDINALIDAD_MAXIMA_CERRADA: nunca pasa por la
        # correccion automatica, se listan aqui las 11 corruptas completas.
        "Asalariado de empresa particular con contrato a t�rmino indefinido":
            "Asalariado de empresa particular con contrato a término indefinido",
        "Asalariado de empresa particular con contrato a t�rmino fijo":
            "Asalariado de empresa particular con contrato a término fijo",
        "Asalariado del Gobierno con contrato a t�rmino indefinido":
            "Asalariado del Gobierno con contrato a término indefinido",
        "Asalariado del Gobierno con contrato a t�rmino fijo":
            "Asalariado del Gobierno con contrato a término fijo",
        "Trabajador de su propia finca (en arriendo o aparcer�a)":
            "Trabajador de su propia finca (en arriendo o aparcería)",
        "Trabajador sin remuneraci�n": "Trabajador sin remuneración",
        "Empleado dom�stico": "Empleado doméstico",
        "Jornalero o pe�n": "Jornalero o peón",
        "Patr�n o empleador": "Patrón o empleador",
        "Trabajador familiar sin remuneraci�n": "Trabajador familiar sin remuneración",
        "Jornalero o trabajador por d�as": "Jornalero o trabajador por días",
    },
    "etnia": {
        "Raizal de archipi�lago": "Raizal de archipiélago",
    },
    "nivel_educ": {
        "B�sica secundaria (6 a 13)": "Básica secundaria (6 a 13)",
    },
    "trabajo_padre": {
        "Jornalero o pe�n": "Jornalero o peón",
        "Trabajador de su propia finca o de una finca que ten�a o tiene en arriendo o aparcer�a":
            "Trabajador de su propia finca o de una finca que tenía o tiene en arriendo o aparcería",
        "Nunca ha trabajado o nunca trabaj�": "Nunca ha trabajado o nunca trabajó",
        "Empleado dom�stico": "Empleado doméstico",
        "Patr�n o empleador": "Patrón o empleador",
        "Trabajador familiar sin remuneraci�n": "Trabajador familiar sin remuneración",
    },
    "trabajo_madre": {
        "Nunca ha trabajado o nunca trabaj�": "Nunca ha trabajado o nunca trabajó",
        "Empleado dom�stico": "Empleado doméstico",
        "Trabajador de su propia finca o de una finca que ten�a o tiene en arriendo o aparcer�a":
            "Trabajador de su propia finca o de una finca que tenía o tiene en arriendo o aparcería",
        "Jornalero o pe�n": "Jornalero o peón",
        "Trabajador familiar sin remuneraci�n": "Trabajador familiar sin remuneración",
        "Patr�n o empleador": "Patrón o empleador",
    },
    "nivel_educ_cursa": {
        "B�sica secundaria(6 a 13)": "Básica secundaria(6 a 13)",
    },
    # ─── Resto del pool candidato a benchmark (ola1+ola2), verificado tras
    # el cruce completo contra docs/variable_audit/personas_hogar_construccion.csv ───
    "ocupacion2": {
        "Trabajador familiar sin remuneraci�n": "Trabajador familiar sin remuneración",
        "Jornalero o pe�n": "Jornalero o peón",
        "Trabajador de su propia finca o de una finca que tiene en arriendo o aparcer�a":
            "Trabajador de su propia finca o de una finca que tiene en arriendo o aparcería",
        # truncado en la fuente original; se conserva el truncamiento
        "Trabajador de su propia finca (propia, en arriendo o aparcer�a, etc. Independien":
            "Trabajador de su propia finca (propia, en arriendo o aparcería, etc. Independien",
        "Otro. Cu�l?": "Otro. Cuál?",
    },
    "registro_mercantil": {
        "Si lo tiene pero no lo renov� este a�o": "Si lo tiene pero no lo renovó este año",
        "Si lo tiene y lo renov� este a�o": "Si lo tiene y lo renovó este año",
        "S� lo tiene pero no lo renov� este a�o": "Sí lo tiene pero no lo renovó este año",
        "S� lo tiene y lo renov� este a�o": "Sí lo tiene y lo renovó este año",
    },
    "medio_consiguio": {
        "No necesit� o no recurri� a ning�n medio": "No necesitó o no recurrió a ningún medio",
        # "directaemente" es un typo de la fuente original, se conserva
        "El empleador lo contact� directaemente": "El empleador lo contactó directaemente",
        "El empleador lo contact� directamente": "El empleador lo contactó directamente",
    },
    "t_busco_trab": {
        "Hace m�s de 5 a�os": "Hace más de 5 años",
        "Entre 2 y menos de 5 a�os": "Entre 2 y menos de 5 años",
    },
    "razon_tiene_negocio": {
        "Por tradici�n familiar": "Por tradición familiar",
    },
    "padre_vive": {"Falleci�": "Falleció"},
    "madre_vive": {"Falleci�": "Falleció"},
    "ultima_hosp": {"Accidente de tr�nsito": "Accidente de tránsito"},
    "recibio_beca": {"No recibi� ninguno": "No recibió ninguno"},
    # Columnas Sí/No con filtro previo (solo se pregunta si aplica): la unica
    # categoria presente termina siendo "S�" sin ningun "No"/"Sí" limpio en la
    # misma columna para que la correccion automatica lo intente resolver.
    "ahorro_futuro": {"S�": "Sí"},
    "ahorro_educ": {"S�": "Sí"},
    "ahorro_casa": {"S�": "Sí"},
    "ahorro_carro": {"S�": "Sí"},
    "ahorro_otros_act": {"S�": "Sí"},
    "ahorro_recre": {"S�": "Sí"},
    "ahorro_montar": {"S�": "Sí"},
    "ahorro_otro": {"S�": "Sí"},
    "beca_otro": {"S�": "Sí"},
    "beca_misma_ins": {"S�": "Sí"},
    "beca_emp_pri": {"S�": "Sí"},
    "beca_emp_pub": {"S�": "Sí"},
    "beca_accionsocial": {"S�": "Sí"},
    "beca_prg_gob": {"S�": "Sí"},
    "beca_cajacom": {"S�": "Sí"},
    # vr_ganancia/vr_salario son numericas con un token categorico ("No
    # recibió") mezclado, mismo patron que ZERO_TOKENS en build_ingreso_hogar.py.
    "vr_ganancia": {"No recibi�": "No recibió"},
    "vr_salario": {"No recibi�": "No recibió"},
    # n_empleados: categoria top-coded, no texto libre real.
    "n_empleados": {"50 personas y m�s": "50 personas y más"},
    "razon_noestudia": {
        "Debe encargarse de labores dom�sticas y/o del cuidado de los ni�os, ancianos o discapacitados":
            "Debe encargarse de labores domésticas y/o del cuidado de los niños, ancianos o discapacitados",
        "No quiere estudiar m�s": "No quiere estudiar más",
        "Termin� su ciclo educativo": "Terminó su ciclo educativo",
        "Necesita educaci�n especial": "Necesita educación especial",
        "Porque tuvo hijos, por embarazo o se cas�": "Porque tuvo hijos, por embarazo o se casó",
    },
    "razon_dejo_trab": {
        "Decidi� no trabajar m�s": "Decidió no trabajar más",
        "Cumpli� el ciclo en ese trabajo": "Cumplió el ciclo en ese trabajo",
        "Se pension� / jubil�": "Se pensionó / jubiló",
        "Otra raz�n": "Otra razón",
        "Cierre o reestructuraci�n de la empresa": "Cierre o reestructuración de la empresa",
        "Despido o declaraci�n de insubsistencia": "Despido o declaración de insubsistencia",
        "Le sali� un trabajo mejor": "Le salió un trabajo mejor",
        "Porque tuvo hijos,  por embarazo o porque se cas�": "Porque tuvo hijos,  por embarazo o porque se casó",
        "Se pension� o jubil�": "Se pensionó o jubiló",
    },
    "actividad_ppal": {
        "Trabaj� en forma remunerada por lo menos una hora":
            "Trabajó en forma remunerada por lo menos una hora",
        "No trabaj� pero ten�a un empleo o trabajo de por lo menos una hora":
            "No trabajó pero tenía un empleo o trabajo de por lo menos una hora",
        "Trabaj� como ayudante familiar sin remuneraci�n por lo menos una hora":
            "Trabajó como ayudante familiar sin remuneración por lo menos una hora",
        "Trabaj� por lo menos una hora y busc� trabajo":
            "Trabajó por lo menos una hora y buscó trabajo",
        # Truncado en la fuente original (corta en "gener"); se conserva el
        # truncamiento, solo se corrige el caracter de reemplazo.
        "Trabaj� en forma remunerada por lo menos UNA hora  en una actividad que le gener":
            "Trabajó en forma remunerada por lo menos UNA hora  en una actividad que le generó",
        "Trabaj� por lo menos UNA hora  en una actividad que le gener� alg�n ingreso":
            "Trabajó por lo menos UNA hora  en una actividad que le generó algún ingreso",
    },
}


# ─── Funciones de limpieza ────────────────────────────────────────────────────

def identificar_columnas_afectadas(df: pd.DataFrame) -> tuple[list, list]:
    """
    Separa las columnas de texto con al menos un valor corrupto en
    (vocabulario_cerrado, texto_libre), segun CARDINALIDAD_MAXIMA_CERRADA.
    La cardinalidad se mide sobre los valores tal cual vienen (antes de
    corregir), que es representativa del numero de categorias reales de la
    pregunta salvo en casos de texto libre.
    """
    afectadas = []
    for c in df.select_dtypes(include="object").columns:
        if df[c].astype(str).str.contains(MARCADOR, regex=False, na=False).any():
            afectadas.append(c)

    cardinalidad = df[afectadas].nunique()
    cerradas = cardinalidad[cardinalidad <= CARDINALIDAD_MAXIMA_CERRADA].index.tolist()
    abiertas = cardinalidad[cardinalidad > CARDINALIDAD_MAXIMA_CERRADA].index.tolist()
    return cerradas, abiertas


def _patron_desde_corrupto(valor_corrupto: str) -> re.Pattern:
    """Construye un regex donde cada '�' es un comodin de exactamente 1 caracter."""
    partes = valor_corrupto.split(MARCADOR)
    return re.compile("^" + ".".join(re.escape(p) for p in partes) + "$")


def corregir_vocabulario_cerrado_automatico(
    df: pd.DataFrame, columnas: list
) -> tuple[pd.DataFrame, dict, dict]:
    """
    Para cada columna de vocabulario cerrado, corrige los valores corruptos
    que tienen exactamente un candidato limpio compatible en esa MISMA
    columna (cualquier ola). Si el unico candidato compatible cae en
    {"Si", "Sí"}, se usa "Sí" (grafia estandar de la ELCA). Los valores sin
    match unico NO se tocan.

    Retorna (df_corregido, mapa_aplicado, residual_sin_resolver), donde
    mapa_aplicado y residual_sin_resolver son {columna: {corrupto: limpio}}
    y {columna: [corruptos]} respectivamente, para el reporte.
    """
    df = df.copy()
    mapa_aplicado: dict = {}
    residual: dict = {}

    for c in columnas:
        serie = df[c].astype(str)
        mask_corrupto = serie.str.contains(MARCADOR, regex=False, na=False)
        if not mask_corrupto.any():
            continue
        limpios = set(serie[~mask_corrupto & (serie != "nan")].unique())
        corruptos = serie[mask_corrupto].unique()

        mapa_col = {}
        for corr in corruptos:
            candidatos = [l for l in limpios if _patron_desde_corrupto(corr).match(l)]
            if len(candidatos) == 1:
                mapa_col[corr] = candidatos[0]
            elif set(candidatos) <= {"Si", "Sí"} and candidatos:
                mapa_col[corr] = "Sí"
            else:
                residual.setdefault(c, []).append(corr)

        if mapa_col:
            df[c] = df[c].replace(mapa_col)
            mapa_aplicado[c] = mapa_col

    return df, mapa_aplicado, residual


def aplicar_correcciones_manuales_prioritarias(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica CORRECCIONES_MANUALES_PRIORITARIAS y valida que no quede '�'."""
    df = df.copy()
    for col, mapa in CORRECCIONES_MANUALES_PRIORITARIAS.items():
        df[col] = df[col].replace(mapa)
        residual = df[col].astype(str).str.contains(MARCADOR, regex=False, na=False)
        assert not residual.any(), (
            f"'{col}' sigue con {residual.sum()} valores corruptos tras aplicar "
            "CORRECCIONES_MANUALES_PRIORITARIAS -- faltan casos por agregar al mapa."
        )
    return df


def documentar_residual(
    residual_automatico: dict, columnas_texto_libre: list, df_original: pd.DataFrame
) -> None:
    """
    Guarda un CSV con todas las columnas que quedan con corrupcion sin
    resolver (vocabulario cerrado sin match automatico + texto libre nunca
    tocado), para no tener que redescubrir el problema si se necesitan
    despues.
    """
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    filas = []
    for col, valores in residual_automatico.items():
        if col in CORRECCIONES_MANUALES_PRIORITARIAS:
            continue  # ya resuelto a mano
        filas.append({
            "columna": col,
            "tipo": "vocabulario_cerrado_sin_match",
            "n_valores_residuales": len(valores),
            "ejemplo": valores[0],
        })
    for col in columnas_texto_libre:
        if col in CORRECCIONES_MANUALES_PRIORITARIAS:
            continue  # ya resuelto a mano (cardinalidad > 25 pero cubierto de todas formas)
        n = df_original[col].astype(str).str.contains(MARCADOR, regex=False, na=False).sum()
        filas.append({
            "columna": col,
            "tipo": "texto_libre_no_tocado",
            "n_valores_residuales": n,
            "ejemplo": "",
        })
    pd.DataFrame(filas).sort_values("n_valores_residuales", ascending=False).to_csv(
        DOC_PATH, index=False
    )


def main() -> None:
    print(f"Leyendo: {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    print(f"  Dimensiones de entrada: {df.shape[0]:,} filas x {df.shape[1]:,} columnas")

    cerradas, abiertas = identificar_columnas_afectadas(df)
    print(f"\nColumnas afectadas por '�': {len(cerradas) + len(abiertas)}")
    print(f"  Vocabulario cerrado (<= {CARDINALIDAD_MAXIMA_CERRADA} categorias): {len(cerradas)}")
    print(f"  Texto libre (sin tocar): {len(abiertas)}")

    df, mapa_aplicado, residual = corregir_vocabulario_cerrado_automatico(df, cerradas)
    n_aplicado = sum(len(v) for v in mapa_aplicado.values())
    n_residual = sum(len(v) for v in residual.values())
    print(f"\nCorreccion automatica (match unico + regla Si/Sí):")
    print(f"  Valores corregidos: {n_aplicado} en {len(mapa_aplicado)} columnas")
    print(f"  Valores sin resolver automaticamente: {n_residual} en {len(residual)} columnas")

    df = aplicar_correcciones_manuales_prioritarias(df)
    n_manual = sum(len(v) for v in CORRECCIONES_MANUALES_PRIORITARIAS.values())
    print(f"\nCorreccion manual de variables prioritarias: {n_manual} valores en "
          f"{len(CORRECCIONES_MANUALES_PRIORITARIAS)} columnas ({', '.join(CORRECCIONES_MANUALES_PRIORITARIAS)})")
    print(f"  Validado: 0 valores '�' restantes en esas {len(CORRECCIONES_MANUALES_PRIORITARIAS)} columnas.")

    documentar_residual(residual, abiertas, pd.read_parquet(INPUT_PATH))
    print(f"\nBacklog de columnas no resueltas documentado en: {DOC_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nBase limpia guardada en: {OUTPUT_PATH}")
    print(f"Dimensiones: {df.shape[0]:,} filas x {df.shape[1]:,} columnas")


if __name__ == "__main__":
    main()
