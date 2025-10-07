import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from io import BytesIO
from PIL import Image


def annuity_monthly(loan, annual_rate, years):
    if loan <= 0:
        return 0.0
    r = annual_rate / 12.0
    n = years * 12
    return loan * (r * (1 + r)**n) / ((1 + r)**n - 1)


def amortization_schedule_monthly(loan, annual_rate, years, start_date='2025-01-01', insurance_rate=0.0):
    payment = annuity_monthly(loan, annual_rate, years)
    schedule = []
    balance = loan
    dates = pd.date_range(start=start_date, periods=years*12, freq='MS')
    for d in dates:
        interest = balance * (annual_rate / 12.0)
        principal = payment - interest
        if principal > balance:
            principal = balance
            payment_this = interest + principal
        else:
            payment_this = payment
        balance = balance - principal
        insurance = balance * (insurance_rate / 12.0)
        schedule.append({'Date': d, 'Paiement': payment_this + insurance, 'Interet': interest,
                         'Principal': principal, 'Assurance': insurance, 'CRD': max(balance,0.0)})
        if balance <= 0:
            break
    return pd.DataFrame(schedule)


def calc_tax_annual(taxable_per_part, parts):
    brackets = [11497, 29315, 83823, 180294]
    rates = [0.0, 0.11, 0.30, 0.41, 0.45]
    tax = 0.0
    lower = 0.0
    for i, upper in enumerate(brackets + [float('inf')]):
        rate = rates[i]
        if taxable_per_part > upper:
            tax += (upper - lower) * rate
            lower = upper
        else:
            tax += (taxable_per_part - lower) * rate
            break
    return tax * parts


def monthly_income_tax(brut1, brut2, parts=2.0, cotisation_rate=0.77):
    net1 = brut1 * cotisation_rate
    net2 = brut2 * cotisation_rate
    taxable_total = net1 + net2
    taxable_per_part = taxable_total / parts

    annual_tax = calc_tax_annual(taxable_per_part, parts)
    revenu_net_apres_impot_annuel = taxable_total - annual_tax

    return revenu_net_apres_impot_annuel / 12.0


def df_to_excel_bytes(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=True, sheet_name='Projection')
    return output.getvalue()


st.set_page_config(layout="wide", page_title="Model Patrimonial",
                   page_icon=r"/Users/ayoub/PycharmProjects/WealthManagementTools/Ressources/icone_patrimoine.png")

cols = st.columns(10)
with cols[0]:
    image = Image.open(r"/Users/ayoub/PycharmProjects/WealthManagementTools/Ressources/icone_patrimoine.png")
    small_image = image.resize((50, 50))
    st.image(image, use_container_width=True)

cols = st.columns(4)
st.header("🏠 Patrimoine du Foyer")
with st.expander("Paramètres", expanded=True):
    st.subheader("Foyer")
    col1, col2, col3 = st.columns(3)
    with col1:
        gross1 = st.number_input("Salaire brut annuel - Personne 1 (€)", value=60000, step=1000)
    with col2:
        gross2 = st.number_input("Salaire brut annuel - Personne 2 (€)", value=38000, step=1000)
    with col3:
        annual_salary_growth = st.number_input("Croissance salariale annuelle (%)", value=5., step=0.01)/100

    col1, col2 = st.columns(2)
    with col1:
        gross_to_net_factor = st.slider("Taux net / brut",0.,100.,77.,0.1)/100
    with col2:
        household_parts = st.number_input("Nombre de parts fiscales", value=2., step=0.5)

    st.markdown("---")
    st.subheader("Épargne et stratégie")
    cols = st.columns(5)
    with cols[0]:
        start_date = st.date_input("Lancement de la stratégie", "2025-09-01")
        end_date = st.date_input("Date de fin", "2060-09-01", min_value=start_date, max_value="2060-09-01")
        st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        savings_rate = st.slider("Taux d'épargne du revenu net (%)", 0, 100, 35)/100
    with col2:
        priority_fraction_to_apport = st.slider("Priorité épargne vers apport (%)", 0, 100, 80) / 100
    col1, col2, col3, col4 = st.columns(4)
    with col2:
        pea_monthly = st.number_input("Versement mensuel PEA (%)", value=50., max_value=100., step=0.01) / 100
        pea_fees = st.number_input("Frais Annuel PEA (%)", value=0.2, max_value=100., step=0.01) / 100
        pea_cap_total = st.number_input("Capacité totale PEA (€)", value=150_000., step=1.)
    with col4:
        av_monthly = st.number_input("Versement mensuel AV (€) [⚠️ post-cash-pea]", value=50., max_value=100., step=0.01) / 100
        av_fees = st.number_input("Frais Annuel AV (%)", value=0.8, max_value=100., step=0.01) / 100
    with col3:
        cto_monthly = st.number_input("Versement mensuel CTO (€) [⚠️ post-cash-pea]", value=50., max_value=100., step=0.01) / 100
        cto_fees = st.number_input("Frais Annuel CTO (%)", value=0.2, max_value=100., step=0.01) / 100
    with col1:
        cash_monthly = st.number_input("Versement mensuel Cash (%)", value=50., max_value=100., step=0.01) / 100
        livretA_rate = st.number_input("Taux Livret A (%)", value=3., step=0.01) / 100
        initial_cash = st.number_input("Cash initial (€)", value=0., step=1000.)
        livretA_cap_total = st.number_input("Capacité totale Livret A (€)", value=2 * 22950., step=1.)

    st.markdown("---")
    st.subheader("Immobilier")
    col1, col2 = st.columns(2)
    with col1:
        property_price = st.number_input("Prix bien (€)", value=500_000, step=10_000)
        loan_years = st.number_input("Durée prêt (ans)", value=25, step=1)
    with col2:
        apport_target = st.number_input("Apport cible (€)", value=50_000, step=5_000)
        annual_mortgage_rate = st.number_input("Taux prêt (%)", value=3., step=0.01)/100
    st.markdown('---')
    col1, col2, col3 = st.columns(3)
    with col1:
        loan_insurance_rate = st.number_input("Taux assurance prêt (%)", value=0.3, step=0.01)/100
    with col2:
        notary_rate = st.number_input("Frais notaire (%)", value=8., step=0.1)/100
    with col3:
        agency_rate = st.number_input("Frais agence (%)", value=2., step=0.1)/100

    st.markdown("---")
    st.subheader("Hypothèses de marché")
    col1, col2 = st.columns(2)
    with col1:
        equity_mean = st.number_input("Rendement actions (%)", 6.,step=0.01)/100
    with col2:
        bond_mean = st.number_input("Rendement obligations (%)", 2.,step=0.01)/100

if st.button("Run Simulation"):
    dates = pd.date_range(start=start_date, end=end_date, freq='MS')

    df = pd.DataFrame(index=dates, columns=[
        'Foyer Salaire Brut', 'Foyer Salaire Super Net', 'Part Epargne',
        'part_apport', 'part_pea', 'part_av', 'part_cto',
        'Apport', 'PEA', 'AV', 'CTO', 'longterm_balance',
        'property_owned', 'property_value', 'loan_balance', 'loan_monthly_payment', 'Immobilier', 'Patrimoine Net',
        'Reserve Cash'
    ], dtype=float).fillna(0.0)

    apport = 0.0
    pea = 0.0
    av = 0.0
    cto = 0.0
    cash_reserve = initial_cash
    loan_active = False
    loan_schedule = None
    total_acquisition_cost = property_price*(1+notary_rate+agency_rate)

    for i, d in enumerate(dates):
        gross1 *= (1+annual_salary_growth) if i>0 and d.month==1 else 1
        gross2 *= (1+annual_salary_growth) if i>0 and d.month==1 else 1
        gross_total = (gross1+gross2)/12
        net_after_tax = monthly_income_tax(gross1, gross2, parts=household_parts, cotisation_rate=gross_to_net_factor)
        monthly_savings = net_after_tax*savings_rate

        if not loan_active and apport < apport_target:
            to_apport = monthly_savings*priority_fraction_to_apport
            remainder = monthly_savings - to_apport
        else:
            to_apport = 0.0
            remainder = monthly_savings

        mr_equity = (1 + equity_mean) ** (1 / 12) - 1
        mr_bond = (1 + bond_mean) ** (1 / 12) - 1
        mr_cash = (1 + livretA_rate) ** (1 / 12) - 1

        pea_fees_monthly = (1 + pea_fees) ** (1 / 12) - 1
        av_fees_monthly = (1 + av_fees) ** (1 / 12) - 1
        cto_fees_monthly = (1 + cto_fees) ** (1 / 12) - 1

        available_cash_cap = max(livretA_cap_total - cash_reserve, 0.0)
        to_cash = min(remainder*cash_monthly, available_cash_cap)
        cash_reserve = cash_reserve * (1 + mr_cash) + to_cash

        available_pea_cap = max(pea_cap_total - pea, 0.0)
        to_pea = min(remainder*pea_monthly, available_pea_cap)
        pea = pea * (1 + mr_equity - pea_fees_monthly) + to_pea

        if available_cash_cap == 0 and available_pea_cap == 0 and remainder > 0:
            to_cto = remainder * cto_monthly
            to_av = remainder * av_monthly
        else:
            to_cto = 0
            to_av = 0
        cto = cto * (1 + 0.6 * mr_equity + 0.4 * mr_bond - cto_fees_monthly) + to_cto
        av = av * (1 + mr_bond - av_fees_monthly) + to_av

        apport = apport*(1+mr_cash) + to_apport

        if not loan_active and apport >= apport_target:
            loan_amount = total_acquisition_cost - apport
            apport -= apport_target
            loan_schedule = amortization_schedule_monthly(loan_amount,annual_mortgage_rate,loan_years,start_date=d.strftime('%Y-%m-%d'),insurance_rate=loan_insurance_rate)
            loan_active = True

        if loan_active:
            row_sched = loan_schedule[loan_schedule['Date']==d]
            if not row_sched.empty:
                payment = float(row_sched['Paiement'].iloc[0])
                balance_loan = float(row_sched['CRD'].iloc[0])
            else:
                payment = 0.0
                balance_loan = float(loan_schedule['CRD'].iloc[-1])
        else:
            payment = 0.0
            balance_loan = 0.0

        prop_owned = 1.0 if loan_active else 0.0
        prop_val = property_price*(1+0.02)**(i/12.0) if prop_owned else property_price
        equity = prop_val - balance_loan if prop_owned else 0.0
        net_worth = apport + pea + av + cto + equity + cash_reserve

        df.loc[d] = [
            gross_total, net_after_tax, monthly_savings,
            to_apport, to_pea, to_av, to_cto,
            apport, pea, av, cto, pea+av+cto,
            prop_owned, prop_val, balance_loan, payment, equity, net_worth,
            cash_reserve
        ]

    st.subheader("Patrimoine du Foyer")

    df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Net Worth", f"{df['Patrimoine Net'].iloc[-1]:,.0f} €")
    col2.metric("PEA", f"{df['PEA'].iloc[-1]:,.0f} €")
    col3.metric("AV", f"{df['AV'].iloc[-1]:,.0f} €")
    col4.metric("CTO", f"{df['CTO'].iloc[-1]:,.0f} €")
    col5.metric("Cash", f"{df['Reserve Cash'].iloc[-1]:,.0f} €")
    if loan_schedule is not None:
        col6.metric("Real Estate", f"{df['Immobilier'].iloc[-1]:,.0f} €")
    else:
        col6.metric("Apport", f"{df['Immobilier'].iloc[-1]:,.0f} €")

    st.subheader("Evolution du patrimoine")
    fig = go.Figure()
    for col in ['PEA', 'AV', 'CTO', 'Apport', 'Immobilier', 'Reserve Cash']:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, stackgroup='one'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Patrimoine Net'], name='Patrimoine Net', line=dict(color='black', width=3)))
    fig.update_layout(height=500, margin=dict(t=20,l=20,r=20), hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Stratégie")
    st.dataframe(df[[
        'Foyer Salaire Super Net','Part Epargne', 'Apport', 'Reserve Cash',
        'PEA','AV','CTO','Immobilier','Patrimoine Net',
    ]].applymap(lambda x: f"{x:,.2f}"))

    if loan_schedule is not None:
        st.subheader("Prêt Immobilier")
        loan_schedule.set_index('Date', inplace=True)
        st.dataframe(loan_schedule.style.format({"Paiement":"{:.2f}", "Interet":"{:.2f}", "Principal":"{:.2f}",
                                                 "Assurance":"{:.2f}", "CRD":"{:.2f}"}))

    st.subheader("Export")
    dfs_to_export = {
        "Stratégie": df,
    }
    if loan_schedule is not None:
        dfs_to_export["Prêt"] = loan_schedule

    cols = st.columns(len(dfs_to_export) + 12)
    for i, k in enumerate(dfs_to_export):
        with cols[i]:
            st.download_button(f"{k}", data=df_to_excel_bytes(dfs_to_export[k]), file_name="projection_patrimoine.xlsx")
