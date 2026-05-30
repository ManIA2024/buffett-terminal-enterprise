import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import logging
from typing import Tuple, Dict, Any, Optional
import numpy as np

# --- CONFIGURACIÓN DE PRODUCCIÓN Y LOGGING ---
st.set_page_config(page_title="Buffett Terminal Enterprise", page_icon="🏦", layout="wide")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- SISTEMA DE CACHÉ PARA EVITAR BANEOS DE LA API ---
@st.cache_data(ttl=86400)
def fetch_financial_data(ticker_symbol: str) -> Tuple[bool, Dict[str, Any]]:
    """Descarga datos financieros y los almacena en caché para optimizar el rendimiento."""
    try:
        logging.info(f"Descargando datos de la API para: {ticker_symbol}")
        ticker = yf.Ticker(ticker_symbol)
        
        info = ticker.info
        if not info or 'currentPrice' not in info:
            if 'regularMarketPrice' in info:
                info['currentPrice'] = info['regularMarketPrice']
            else:
                return False, {"error": "Ticker inválido o sin datos de precio actual."}

        data = {
            "info": info,
            "financials": ticker.financials,
            "cashflow": ticker.cashflow,
            "balance_sheet": ticker.balance_sheet
        }
        return True, data
    except Exception as e:
        logging.error(f"Error fetching data for {ticker_symbol}: {str(e)}")
        return False, {"error": f"Error de conexión: {str(e)}"}

# --- MOTOR MATEMÁTICO (CLASE AISLADA DE LA UI) ---
class ValuationEngine:
    def __init__(self, data: Dict[str, Any], discount_rate: float, margin_of_safety: float, max_growth: float):
        self.data = data
        self.info = data['info']
        self.financials = data['financials']
        self.cashflow = data['cashflow']
        self.balance_sheet = data['balance_sheet']
        
        self.current_price = self.info.get('currentPrice', 0)
        self.shares_outstanding = self.info.get('sharesOutstanding', 1)
        
        self.discount_rate = discount_rate
        self.margin_of_safety = margin_of_safety
        self.max_growth_rate = max_growth
        self.terminal_multiple = 12

    def calculate_owner_earnings(self) -> float:
        """Calcula el Flujo de Caja Libre / Owner Earnings con múltiples fallbacks de seguridad."""
        try:
            if 'Net Income' in self.financials.index:
                net_income = self.financials.loc['Net Income'].iloc[0]
            else:
                net_income = self.info.get('netIncomeToCommon', 0)

            depreciation = 0
            if 'Depreciation And Amortization' in self.cashflow.index:
                depreciation = self.cashflow.loc['Depreciation And Amortization'].iloc[0]
                
            capex = 0
            if 'Capital Expenditure' in self.cashflow.index:
                capex = abs(self.cashflow.loc['Capital Expenditure'].iloc[0])

            owner_earnings = net_income + depreciation - capex
            
            if pd.isna(owner_earnings) or owner_earnings <= 0:
                if 'Free Cash Flow' in self.cashflow.index:
                    fcf = self.cashflow.loc['Free Cash Flow'].iloc[0]
                    return fcf if not pd.isna(fcf) else 0
                 
            return owner_earnings
        except Exception as e:
            logging.warning(f"Error en Owner Earnings: {e}. Usando fallback de FCF.")
            try:
                fcf = self.cashflow.loc['Free Cash Flow'].iloc[0]
                return fcf if not pd.isna(fcf) else 0
            except:
                return 0

    def calculate_dcf(self, custom_discount: Optional[float] = None, custom_growth: Optional[float] = None) -> Dict[str, float]:
        """Calcula el modelo de flujos de caja descontados. Permite parámetros custom para la matriz de sensibilidad."""
        dr = custom_discount if custom_discount is not None else self.discount_rate
        gr = custom_growth if custom_growth is not None else min(self.info.get('earningsGrowth', 0.05), self.max_growth_rate)
        
        if gr <= 0: 
            gr = 0.03
             
        base_cashflow = self.calculate_owner_earnings()
        if base_cashflow <= 0:
            return {"intrinsic_value": 0, "buy_price": 0, "status": "Flujo Negativo"}

        present_value_cf = 0
        current_cf = base_cashflow
        
        for year in range(1, 11):
            current_cf *= (1 + gr)
            present_value_cf += current_cf / ((1 + dr) ** year)

        terminal_value = (current_cf * self.terminal_multiple) / ((1 + dr) ** 10)
        total_value = present_value_cf + terminal_value
        
        iv_per_share = total_value / self.shares_outstanding
        buy_price = iv_per_share * (1 - self.margin_of_safety)
        
        return {"intrinsic_value": iv_per_share, "buy_price": buy_price, "growth_used": gr, "base_cashflow": base_cashflow}

    def calculate_piotroski_f_score(self) -> Dict[str, Any]:
        """Calcula el Piotroski F-Score (0-9) comparando el año actual vs el año anterior."""
        try:
            if len(self.financials.columns) < 2 or len(self.balance_sheet.columns) < 2 or len(self.cashflow.columns) < 2:
                return {"score": "N/A", "error": "Datos históricos insuficientes (se requieren 2 años)."}

            fin_0, fin_1 = self.financials.iloc[:, 0], self.financials.iloc[:, 1]
            bs_0, bs_1 = self.balance_sheet.iloc[:, 0], self.balance_sheet.iloc[:, 1]
            cf_0 = self.cashflow.iloc[:, 0]

            def get_val(series, keys, default=0):
                for key in keys:
                    if key in series.index and not pd.isna(series[key]):
                        return series[key]
                return default

            score = 0
            breakdown = {}

            ni_0 = get_val(fin_0, ['Net Income', 'Net Income Common Stockholders'])
            ni_1 = get_val(fin_1, ['Net Income', 'Net Income Common Stockholders'])
            assets_0 = get_val(bs_0, ['Total Assets'], 1)
            assets_1 = get_val(bs_1, ['Total Assets'], 1)
            roa_0 = ni_0 / assets_0
            roa_1 = ni_1 / assets_1
            cfo_0 = get_val(cf_0, ['Operating Cash Flow', 'Total Cash From Operating Activities'])

            if roa_0 > 0: 
                score += 1
                breakdown['1. ROA Positivo'] = "✅ Sí"
            else: 
                breakdown['1. ROA Positivo'] = "❌ No"
            
            if cfo_0 > 0: 
                score += 1
                breakdown['2. Flujo Caja Operativo > 0'] = "✅ Sí"
            else: 
                breakdown['2. Flujo Caja Operativo > 0'] = "❌ No"
            
            if roa_0 > roa_1: 
                score += 1
                breakdown['3. Crecimiento del ROA'] = "✅ Sí"
            else: 
                breakdown['3. Crecimiento del ROA'] = "❌ No"
            
            if cfo_0 > ni_0: 
                score += 1
                breakdown['4. CFO > Utilidad Neta'] = "✅ Sí"
            else: 
                breakdown['4. CFO > Utilidad Neta'] = "❌ No"

            debt_0 = get_val(bs_0, ['Long Term Debt', 'Total Debt'])
            debt_1 = get_val(bs_1, ['Long Term Debt', 'Total Debt'])
            if (debt_0 / assets_0) < (debt_1 / assets_1): 
                score += 1
                breakdown['5. Reducción de Deuda'] = "✅ Sí"
            else: 
                breakdown['5. Reducción de Deuda'] = "❌ No"

            ca_0 = get_val(bs_0, ['Current Assets'])
            cl_0 = get_val(bs_0, ['Current Liabilities'], 1)
            ca_1 = get_val(bs_1, ['Current Assets'])
            cl_1 = get_val(bs_1, ['Current Liabilities'], 1)
            if (ca_0 / cl_0) > (ca_1 / cl_1): 
                score += 1
                breakdown['6. Mejora de Liquidez'] = "✅ Sí"
            else: 
                breakdown['6. Mejora de Liquidez'] = "❌ No"

            shares_0 = get_val(bs_0, ['Ordinary Shares Number', 'Share Issued'])
            shares_1 = get_val(bs_1, ['Ordinary Shares Number', 'Share Issued'])
            if shares_0 <= shares_1: 
                score += 1
                breakdown['7. Sin Dilución de Acciones'] = "✅ Sí"
            else: 
                breakdown['7. Sin Dilución de Acciones'] = "❌ No"

            gp_0 = get_val(fin_0, ['Gross Profit'])
            rev_0 = get_val(fin_0, ['Total Revenue'], 1)
            gp_1 = get_val(fin_1, ['Gross Profit'])
            rev_1 = get_val(fin_1, ['Total Revenue'], 1)
            if (gp_0 / rev_0) > (gp_1 / rev_1): 
                score += 1
                breakdown['8. Mejora Margen Bruto'] = "✅ Sí"
            else: 
                breakdown['8. Mejora Margen Bruto'] = "❌ No"

            if (rev_0 / assets_0) > (rev_1 / assets_1): 
                score += 1
                breakdown['9. Mejora Rotación Activos'] = "✅ Sí"
            else: 
                breakdown['9. Mejora Rotación Activos'] = "❌ No"

            return {"score": score, "breakdown": breakdown}

        except Exception as e:
            logging.error(f"Error calculando Piotroski para {self.info.get('symbol')}: {e}")
            return {"score": "Error", "error": str(e)}

    def get_multiples_data(self) -> Dict[str, Any]:
        """Extrae los múltiplos actuales y las métricas base necesarias para la valoración relativa."""
        info = self.info
        
        # Extracción de métricas base (por acción y totales)
        eps = info.get('trailingEps', 0)
        book_value_ps = info.get('bookValue', 0)
        revenue_ps = info.get('revenuePerShare', 0)
        
        ebitda = info.get('ebitda', 0)
        fcf = info.get('freeCashflow', 0) # NUEVO: Flujo de Caja Libre
        total_cash = info.get('totalCash', 0)
        total_debt = info.get('totalDebt', 0)
        enterprise_value = info.get('enterpriseValue', 0)
        
        # Extracción de los múltiplos actuales con los que cotiza la empresa
        current_pe = info.get('trailingPE', 0)
        current_pb = info.get('priceToBook', 0)
        current_ps = info.get('priceToSalesTrailing12Months', 0)
        current_ev_ebitda = info.get('enterpriseToEbitda', 0)
        
        # NUEVO: Calcular EV/FCF actual (protegiendo contra división por cero o flujos negativos)
        current_ev_fcf = 0
        if fcf and fcf > 0 and enterprise_value:
            current_ev_fcf = enterprise_value / fcf

        return {
            "eps": eps,
            "book_value_ps": book_value_ps,
            "revenue_ps": revenue_ps,
            "ebitda": ebitda,
            "fcf": fcf, # Añadido al diccionario
            "total_cash": total_cash,
            "total_debt": total_debt,
            "shares": self.shares_outstanding,
            "current_pe": current_pe,
            "current_pb": current_pb,
            "current_ps": current_ps,
            "current_ev_ebitda": current_ev_ebitda,
            "current_ev_fcf": current_ev_fcf # Añadido al diccionario
        }

    def run_monte_carlo(self, iterations: int = 5000) -> Optional[np.ndarray]:
        """Ejecuta miles de simulaciones DCF variando el crecimiento y el WACC."""
        try:
            base_cf = self.calculate_owner_earnings()
            if base_cf <= 0: return None

            base_growth = min(self.info.get('earningsGrowth', 0.05), self.max_growth_rate)
            if base_growth <= 0: base_growth = 0.03

            # Generar distribuciones normales (Campana de Gauss)
            # WACC varía +/- 1.5% en promedio. Crecimiento varía +/- 2.5%
            wacc_dist = np.random.normal(loc=self.discount_rate, scale=0.015, size=iterations)
            growth_dist = np.random.normal(loc=base_growth, scale=0.025, size=iterations)

            results = []
            for i in range(iterations):
                wacc = max(0.04, wacc_dist[i]) # El WACC nunca debe bajar del 4% (tasa libre de riesgo)
                g = growth_dist[i]
                
                cf = base_cf
                pv = 0
                for year in range(1, 11):
                    cf *= (1 + g)
                    pv += cf / ((1 + wacc) ** year)
                
                tv = (cf * self.terminal_multiple) / ((1 + wacc) ** 10)
                intrinsic_value = (pv + tv) / self.shares_outstanding
                results.append(intrinsic_value)

            return np.array(results)
        except Exception as e:
            logging.error(f"Error en Montecarlo: {e}")
            return None


# --- INTERFAZ DE USUARIO (UI) ---
st.title("🏦 Terminal Institucional de Value Investing")
st.markdown("Plataforma de valoración automatizada basada en Flujos de Caja Descontados y Moat Financiero.")

# Barra lateral
with st.sidebar:
    st.header("⚙️ Parámetros del Modelo")
    ticker_input = st.text_input("Ticker de cotización:", value="AAPL").upper().strip()
    
    with st.expander("Ajustes del DCF", expanded=True):
        discount_input = st.slider("Tasa de Descuento (WACC estimado) %", 5.0, 20.0, 10.0, step=0.5) / 100
        margin_input = st.slider("Margen de Seguridad %", 0, 50, 30, step=5) / 100
        growth_input = st.slider("Crecimiento Máximo Aceptado %", 2.0, 25.0, 15.0, step=0.5) / 100
        
    if st.button("Limpiar Caché 🔄"):
        st.cache_data.clear()
        st.success("Caché borrada. Los próximos datos serán descargados en vivo.")

if ticker_input:
    with st.spinner("Extrayendo y procesando datos financieros..."):
        success, raw_data = fetch_financial_data(ticker_input)
        
    if not success:
        st.error(raw_data.get("error", "Error desconocido."))
    else:
        engine = ValuationEngine(raw_data, discount_input, margin_input, growth_input)
        info = engine.info
        
        # --- HEADER ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Precio Actual", f"${engine.current_price:.2f}")
        col2.metric("Industria", info.get('industry', 'N/A'))
        col3.metric("PER (TTM)", f"{info.get('trailingPE', 0):.2f}")
        col4.metric("ROE", f"{info.get('returnOnEquity', 0)*100:.2f}%")

        st.divider()

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["💰 Valoración Principal", "📊 Matriz de Sensibilidad", "📑 Salud Financiera", "⚖️ Múltiplos", "🆚 Comparador", "🎲 Montecarlo"])

        # --- PESTAÑA 1: DCF PRINCIPAL ---
        with tab1:
            col_v1, col_v2 = st.columns([1, 2])
            val_results = engine.calculate_dcf()
            
            with col_v1:
                st.markdown("### Veredicto del Algoritmo")
                if val_results['intrinsic_value'] == 0:
                    st.error("🚨 La empresa tiene flujos de caja libre negativos o nulos. El modelo DCF no es aplicable.")
                else:
                    iv = val_results['intrinsic_value']
                    bp = val_results['buy_price']
                    cp = engine.current_price
                    
                    st.metric("Valor Intrínseco", f"${iv:.2f}")
                    st.metric("Precio con Margen de Seguridad", f"${bp:.2f}")
                    
                    if cp <= bp:
                        st.success("🟢 **STRONG BUY**: Cotiza con el margen de seguridad requerido.")
                    elif cp <= iv:
                        st.warning("🟡 **HOLD / FAIR VALUE**: Cotiza por debajo de su valor intrínseco, pero sin el margen configurado.")
                    else:
                        st.error("🔴 **OVERVALUED**: El precio de mercado exige expectativas irreales de crecimiento.")

            with col_v2:
                if val_results['intrinsic_value'] > 0:
                    gr = val_results['growth_used']
                    cf = val_results['base_cashflow']
                    years, values = [], []
                    for i in range(1, 11):
                        cf *= (1 + gr)
                        years.append(f"A{i}")
                        values.append(cf)
                        
                    fig_cf = px.bar(x=years, y=values, title="Proyección de Flujos (Sin descontar)", labels={'x': 'Años Futuros', 'y': 'Flujo ($)'})
                    fig_cf.update_traces(marker_color='#2ca02c')
                    st.plotly_chart(fig_cf, use_container_width=True)

        # --- PESTAÑA 2: MATRIZ DE SENSIBILIDAD ---
        with tab2:
            st.markdown("### Matriz de Sensibilidad: Valor Intrínseco")
            st.markdown("¿Cómo cambia el valor real de la acción si nos equivocamos en nuestras proyecciones de crecimiento o tasa de descuento?")
            
            if val_results['intrinsic_value'] > 0:
                dr_range = [discount_input - 0.02, discount_input - 0.01, discount_input, discount_input + 0.01, discount_input + 0.02]
                gr_range = [val_results['growth_used'] - 0.04, val_results['growth_used'] - 0.02, val_results['growth_used'], val_results['growth_used'] + 0.02, val_results['growth_used'] + 0.04]
                
                matrix_data = []
                for dr in dr_range:
                    row = {}
                    for gr in gr_range:
                        if gr < 0: 
                            gr = 0
                        res = engine.calculate_dcf(custom_discount=dr, custom_growth=gr)
                        row[f"Crecimiento {gr*100:.1f}%"] = f"${res['intrinsic_value']:.2f}"
                    matrix_data.append(row)
                
                df_matrix = pd.DataFrame(matrix_data)
                df_matrix.index = [f"Descuento {d*100:.1f}%" for d in dr_range]
                
                st.dataframe(df_matrix, use_container_width=True)
                st.caption("El valor central corresponde a los parámetros configurados en la barra lateral.")

        # --- PESTAÑA 3: SALUD FINANCIERA Y PIOTROSKI ---
        with tab3:
            st.markdown("### 🏰 Análisis de Fortaleza Financiera (Foso Económico)")
            
            piotroski_data = engine.calculate_piotroski_f_score()
            
            c1, c2 = st.columns([1, 2])
            
            with c1:
                st.markdown("#### Piotroski F-Score")
                score = piotroski_data.get('score')
                
                if isinstance(score, int):
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = score,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Calificación sobre 9"},
                        gauge = {
                            'axis': {'range': [None, 9]},
                            'bar': {'color': "black"},
                            'steps': [
                                {'range': [0, 3], 'color': "#ff4b4b"},
                                {'range': [4, 6], 'color': "#ffa600"},
                                {'range': [7, 9], 'color': "#2ca02c"}
                            ]
                        }
                    ))
                    fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    if score >= 7:
                        st.success("🌟 **Fuerte:** La empresa tiene una salud financiera sobresaliente. Ideal para Value Investing.")
                    elif score >= 4:
                        st.warning("⚖️ **Estable:** Fundamentales mixtos. Requiere revisión de las métricas falladas.")
                    else:
                        st.error("🚨 **Débil:** Alto riesgo fundamental. Posible trampa de valor.")
                else:
                    st.error(f"No se pudo calcular el F-Score. {piotroski_data.get('error', '')}")

            with c2:
                if isinstance(score, int):
                    st.markdown("#### Desglose de los 9 Criterios")
                    df_breakdown = pd.DataFrame(list(piotroski_data['breakdown'].items()), columns=['Criterio (Evalúa Año Actual vs Anterior)', 'Aprobado'])
                    st.dataframe(df_breakdown, hide_index=True, use_container_width=True)
                    
            st.divider()
            
            st.markdown("#### Checklist Rápido de Liquidez y Solvencia")
            col_l1, col_l2 = st.columns(2)
            de_ratio = info.get('debtToEquity', 0) / 100
            cr_ratio = info.get('currentRatio', 0)
            
            with col_l1:
                st.info(f"**Deuda / Patrimonio:** {de_ratio:.2f}")
                if 0 <= de_ratio <= 0.5: 
                    st.success("✅ Nivel de deuda excelente y manejable.")
                elif de_ratio <= 1.2: 
                    st.warning("⚠️ Deuda moderada.")
                else: 
                    st.error("❌ Alto apalancamiento financiero.")
                 
            with col_l2:
                st.info(f"**Ratio de Liquidez (Current Ratio):** {cr_ratio:.2f}")
                if cr_ratio >= 1.5: 
                    st.success("✅ Cubre fácilmente sus pasivos a corto plazo.")
                else: 
                    st.error("❌ Liquidez ajustada. Cuidado con problemas de caja a corto plazo.")

# --- PESTAÑA 4: VALORACIÓN POR MÚLTIPLOS ---
        with tab4:
            st.markdown("### ⚖️ Valoración Relativa (Comparativa de Mercado)")
            st.markdown("Observa cómo cotiza la empresa hoy y calcula su Precio Implícito al compararla con el promedio de sus competidores.")
            
            multiples_data = engine.get_multiples_data()
            
            # 1. RENDERS DE MÉTRICAS (Verifica que existan las 5 columnas)
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("PER Actual", f"{multiples_data['current_pe']:.2f}" if multiples_data['current_pe'] else "N/A")
            c2.metric("EV / EBITDA", f"{multiples_data['current_ev_ebitda']:.2f}" if multiples_data['current_ev_ebitda'] else "N/A")
            
            # Verificación segura del EV/FCF
            ev_fcf_val = multiples_data.get('current_ev_fcf', 0)
            c3.metric("EV / FCF", f"{ev_fcf_val:.2f}" if ev_fcf_val > 0 else "N/A")
            
            c4.metric("Price / Book", f"{multiples_data['current_pb']:.2f}" if multiples_data['current_pb'] else "N/A")
            c5.metric("Price / Sales", f"{multiples_data['current_ps']:.2f}" if multiples_data['current_ps'] else "N/A")
            
            st.divider()
            
            # 2. CALCULADORA INTERACTIVA
            st.markdown("#### 🧮 Calculadora de Precio Implícito")
            col_sel, col_val, col_res = st.columns(3)
            
            with col_sel:
                metodo = st.selectbox(
                    "1. Selecciona el Múltiplo a evaluar:", 
                    ["PER (Price/Earnings)", "EV / EBITDA", "EV / FCF", "P/B (Price / Book)", "P/S (Price / Sales)"]
                )
                
            with col_val:
                target_multiple = st.number_input(
                    "2. Ingresa el Múltiplo Objetivo (Promedio Sector):", 
                    min_value=0.1, 
                    value=15.0, 
                    step=0.5
                )
                
            with col_res:
                st.markdown("#### 🎯 Precio Implícito Estimado")
                implied_price = 0
                
                # EJECUCIÓN DEL MOTOR MATEMÁTICO
                if metodo == "PER (Price/Earnings)":
                    implied_price = multiples_data['eps'] * target_multiple
                    st.caption("Fórmula: EPS × PER Objetivo")
                    
                elif metodo == "P/B (Price / Book)":
                    implied_price = multiples_data['book_value_ps'] * target_multiple
                    st.caption("Fórmula: Valor en Libros por Acción × P/B Objetivo")
                    
                elif metodo == "P/S (Price / Sales)":
                    implied_price = multiples_data['revenue_ps'] * target_multiple
                    st.caption("Fórmula: Ventas por Acción × P/S Objetivo")
                    
                elif metodo == "EV / EBITDA":
                    if multiples_data['ebitda'] and multiples_data['shares']:
                        target_ev = multiples_data['ebitda'] * target_multiple
                        implied_equity_value = target_ev + multiples_data['total_cash'] - multiples_data['total_debt']
                        implied_price = implied_equity_value / multiples_data['shares']
                    st.caption("Fórmula: ((EBITDA × Múltiplo) + Caja - Deuda) ÷ Acciones")

                elif metodo == "EV / FCF":
                    fcf_val = multiples_data.get('fcf', 0)
                    if fcf_val and fcf_val > 0 and multiples_data['shares']:
                        target_ev = fcf_val * target_multiple
                        implied_equity_value = target_ev + multiples_data['total_cash'] - multiples_data['total_debt']
                        implied_price = implied_equity_value / multiples_data['shares']
                    st.caption("Fórmula: ((FCF × Múltiplo) + Caja - Deuda) ÷ Acciones")

                # DESPLIEGUE DEL RESULTADO FINAL
                if implied_price > 0:
                    diferencia_porcentual = (implied_price / engine.current_price - 1) * 100
                    st.metric(
                        "Valor Justo Calculado", 
                        f"${implied_price:.2f}", 
                        delta=f"{diferencia_porcentual:.1f}% vs Mercado"
                    )
                else:
                    st.error("Datos insuficientes o flujos negativos para la métrica seleccionada.")
                    
# --- PESTAÑA 5: COMPARADOR DE EMPRESAS (PEER ANALYSIS) ---
        with tab5:
            st.markdown("### 🆚 Comparador de Competidores")
            st.markdown("Enfrenta a tu empresa principal contra un rival directo. El sistema otorgará un 🏆 a la métrica más favorable (ej. Mayor rentabilidad o menor nivel de deuda).")
            
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.info(f"**Empresa Principal:** {info.get('shortName', ticker_input)} ({ticker_input})")
            with col_c2:
                ticker_peer = st.text_input("Ingresa el Ticker del Competidor (Ej: AMD para INTC, o KO para PEP):", value="").upper().strip()

            if ticker_peer:
                if ticker_peer == ticker_input:
                    st.warning("⚠️ Ingresa un ticker diferente para comparar.")
                else:
                    with st.spinner(f"Analizando métricas de {ticker_peer}..."):
                        success_peer, raw_data_peer = fetch_financial_data(ticker_peer)

                    if not success_peer:
                        st.error(f"Error al extraer datos de {ticker_peer}: {raw_data_peer.get('error')}")
                    else:
                        info_peer = raw_data_peer['info']

                        # Definir métricas a comparar: (Nombre, Valor 1, Valor 2, ¿Mayor es mejor?)
                        comparisons = [
                            ("Capitalización de Mercado", info.get('marketCap', 0), info_peer.get('marketCap', 0), True),
                            ("PER (Price to Earnings)", info.get('trailingPE', 0), info_peer.get('trailingPE', 0), False),
                            ("Price / Book (P/B)", info.get('priceToBook', 0), info_peer.get('priceToBook', 0), False),
                            ("EV / EBITDA", info.get('enterpriseToEbitda', 0), info_peer.get('enterpriseToEbitda', 0), False),
                            ("ROE (Retorno sobre Capital) %", info.get('returnOnEquity', 0)*100, info_peer.get('returnOnEquity', 0)*100, True),
                            ("Margen Bruto %", info.get('grossMargins', 0)*100, info_peer.get('grossMargins', 0)*100, True),
                            ("Margen Neto %", info.get('profitMargins', 0)*100, info_peer.get('profitMargins', 0)*100, True),
                            ("Deuda / Capital (Menor es mejor)", info.get('debtToEquity', 0)/100, info_peer.get('debtToEquity', 0)/100, False),
                            ("Ratio de Liquidez (Current Ratio)", info.get('currentRatio', 0), info_peer.get('currentRatio', 0), True)
                        ]

                        table_data = []
                        for name, val1, val2, higher_is_better in comparisons:
                            # Limpiar N/As o None
                            v1 = val1 if pd.notna(val1) and val1 is not None else 0
                            v2 = val2 if pd.notna(val2) and val2 is not None else 0

                            # Lógica del ganador (Ignorando ceros si la métrica es de valuación donde 0 = sin datos)
                            w1, w2 = "", ""
                            if v1 != v2:
                                if higher_is_better:
                                    if v1 > v2: w1 = "🏆"
                                    elif v2 > v1: w2 = "🏆"
                                else:
                                    # Para PER, P/B, etc., menor es mejor, pero NO si es 0 o negativo (que indica pérdida o falta de datos)
                                    if (0 < v1 < v2) or (v1 > 0 and v2 <= 0): w1 = "🏆"
                                    elif (0 < v2 < v1) or (v2 > 0 and v1 <= 0): w2 = "🏆"

                            # Formatear el texto a mostrar
                            def format_val(val, metric_name):
                                if val == 0: return "N/A"
                                if "Capitalización" in metric_name: return f"${val/1e9:.2f}B"
                                if "%" in metric_name: return f"{val:.2f}%"
                                return f"{val:.2f}"

                            str_v1 = f"{w1} {format_val(v1, name)}".strip()
                            str_v2 = f"{w2} {format_val(v2, name)}".strip()

                            table_data.append({
                                "Métrica Financiera": name,
                                f"{ticker_input}": str_v1,
                                f"{ticker_peer}": str_v2
                            })

                        # Renderizar Tabla
                        st.markdown(f"#### 📊 {ticker_input} vs {ticker_peer}")
                        df_compare = pd.DataFrame(table_data)
                        
                        # Mostramos el DataFrame limpio y expandido
                        st.dataframe(df_compare, hide_index=True, use_container_width=True)
                        st.caption("*Nota: En múltiplos de valoración (PER, EV/EBITDA), las empresas con ganancias negativas o nulas aparecen como N/A.*")                    

# --- PESTAÑA 6: SIMULACIÓN DE MONTECARLO ---
        with tab6:
            st.markdown("### 🎲 Simulación Estocástica (Montecarlo)")
            st.markdown("El modelo tradicional calcula 1 solo futuro. Este algoritmo calcula **5,000 futuros posibles** variando estadísticamente el crecimiento y las tasas de interés para mostrarte las probabilidades reales.")
            
            if st.button("▶️ Ejecutar 5,000 Simulaciones", type="primary"):
                with st.spinner("Generando universos paralelos de flujo de caja..."):
                    mc_results = engine.run_monte_carlo(iterations=5000)
                    
                if mc_results is not None:
                    # Limpiar datos atípicos extremos (outliers) para un mejor gráfico
                    q_low = np.percentile(mc_results, 2)
                    q_high = np.percentile(mc_results, 98)
                    mc_filtered = mc_results[(mc_results > q_low) & (mc_results < q_high)]
                    
                    # Calcular Percentiles clave
                    p20 = np.percentile(mc_filtered, 20) # Caso Pesimista
                    p50 = np.percentile(mc_filtered, 50) # Caso Base (Mediana)
                    p80 = np.percentile(mc_filtered, 80) # Caso Optimista
                    
                    # Crear Histograma
                    fig_mc = px.histogram(
                        x=mc_filtered, 
                        nbins=60, 
                        title="Distribución de Probabilidad del Valor Intrínseco",
                        labels={'x': 'Valor Intrínseco Estimado ($)', 'y': 'Frecuencia (Escenarios)'},
                        color_discrete_sequence=['#1f77b4']
                    )
                    
                    # Añadir líneas de referencia
                    fig_mc.add_vline(x=engine.current_price, line_dash="dash", line_color="red", annotation_text="Precio Actual Mercado")
                    fig_mc.add_vline(x=p50, line_dash="solid", line_color="green", annotation_text="Caso Base (50%)")
                    
                    st.plotly_chart(fig_mc, use_container_width=True)
                    
                    # Mostrar Resumen Estadístico
                    c_pes, c_base, c_opt = st.columns(3)
                    c_pes.metric("🔴 Caso Pesimista (Peor 20%)", f"${p20:.2f}")
                    c_base.metric("⚖️ Caso Base (Mediana)", f"${p50:.2f}")
                    c_opt.metric("🟢 Caso Optimista (Mejor 20%)", f"${p80:.2f}")
                    
                    st.info(f"💡 **Interpretación:** En el 80% de los 5,000 escenarios simulados, la acción vale al menos **${p20:.2f}**. Si el precio actual (${engine.current_price:.2f}) está cerca o por debajo de esta línea roja, el riesgo de pérdida a largo plazo es matemáticamente muy bajo.")
                else:
                    st.error("No se pudo correr la simulación (Flujos negativos o falta de datos).")