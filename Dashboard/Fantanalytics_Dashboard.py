import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, accuracy_score
import altair as alt
import streamlit as st

def setup():
  PAGE_CONFIG = {"page_title":"FantAnalytics",
                  "page_icon":":bar_chart:",
                  "layout":"wide",
                  "theme":"dark"}
  st.set_page_config(**PAGE_CONFIG)

  setup()

df = pd.read_csv("./Predizioni/2022-23.csv", index_col=0)

################################################################################

def accuracy_sign(y_val, preds, previous_period, given_threshold:float=None, absolute_error:bool=False):
    '''
    Compute accuracy of predicted trend w.r.t. previous_period
    
    given_threshold: meant to avoid distortions in the absence of any trend. If provided, predictions with error < given_threshold are considered correct, regardless of the actual sign
    absolute_error: ignored when given_threshold is None. If False, given_threshold is evaluated as % error, else as an abolute_error
    '''

    # Make sure we are dealing with three Series with the same index
    y_val = pd.Series(y_val).reset_index(drop=True)
    preds = pd.Series(preds).reset_index(drop=True)
    previous_period = pd.Series(previous_period).reset_index(drop=True)
    index = [i for i in y_val.index]

    # For each sample, compute the real and the predicted trends
    real = [1 if (y_val[i] - previous_period[i]) >= 0 else 0 for i in index]
    predicted = [1 if (preds[i] - previous_period[i]) >= 0 else 0 for i in index]
    
    # Modify predicted to match real when the error is neglectable
    if given_threshold is not None:
        # Compute errors
        if not absolute_error:
            # Add very small value when zero to make division possible
            # Error is below threshold if the predicted value is exactly zero as well
            y_val = [x if x != 0 else .0001 for x in y_val]
            preds = [x if x != 0 else .0001 for x in preds]
            ratios = [preds[i] / y_val[i] for i in range(size)]
            error_below_threshold = [(x > 1 - given_threshold) and (x < 1 + given_threshold) for x in ratios]
        else:
            error_below_threshold = [abs(preds[i] - y_val[i]) < given_threshold for i in index]

        real = [real[i] for i in range(len(predicted)) if error_below_threshold[i] == False]
        predicted = [predicted[i] for i in range(len(predicted)) if error_below_threshold[i] == False]

    return accuracy_score(real, predicted)

################################################################################

def scatter(df, x:str, y:str, tooltip:list=["Nome"], x_min=None, x_max=None, y_min=None, y_max=None, allinea_assi:bool=False, diagonali:bool=False):

    df = df.reset_index().round(2)

    x_min = df[x].min() if x_min is None else x_min
    x_max = df[x].max() if x_max is None else x_max
    y_min = df[y].min() if y_min is None else y_min
    y_max = df[y].max() if y_max is None else y_max
    axis_min = min([x_min, y_min])
    axis_max = max([x_max, y_max])

    x_domain = [x_min-.5, x_max+.5] if not allinea_assi else [axis_min-.5, axis_max+.5]
    y_domain = [y_min-.5, y_max+.5] if not allinea_assi else [axis_min-.5, axis_max+.5]

    graph = (alt.Chart(df)
        .mark_circle(size=40)
        .encode(
            x=alt.X(x, scale=alt.Scale(domain=x_domain), title=x),
            y=alt.Y(y, scale=alt.Scale(domain=y_domain), title=y),
            color=alt.Color("Ruolo Anno Successivo", scale=alt.Scale(domain=["P", "D", "C", "A"], range=["#F8AB12", "#65C723", "#136AF6", "#F21C3C"]), title="Ruolo"),
            tooltip=alt.Tooltip(tooltip))
        .interactive())
    
    if diagonali:

        central_line = (alt.Chart(pd.DataFrame({'var1': [-100, 100], 'var2': [-100, 100]}))
            .mark_line()
            .encode(
                x=alt.X('var1', title=""),
                y=alt.Y('var2', title=""))
            .interactive())
        
        steps = [i for i in list(np.arange(-1, 1.1, .125)) if i != 0]
        counter = 0
        for step in steps:
            globals()["line"+str(counter)] = (alt.Chart(pd.DataFrame(pd.DataFrame({'var1': [-100-step, 100-step], 'var2': [-100+step, 100+step]})))
            .mark_line(color="grey", opacity=(1-abs(step))/2)
            .encode(
                x=alt.X('var1', title=""),
                y=alt.Y('var2', title=""))
                )
            counter +=1

        return graph + central_line + alt.layer(*(globals()["line"+str(i)] for i in range(counter)))
    
    return graph

def bars_multiple_features(df, x:list, y:str, x_name:str=" ", y_name:str=" ", x_min=None, x_max=None):

    x_min = min([df[i].min() for i in x]) if x_min is None else x_min
    x_max = max([df[i].max() for i in x]) if x_max is None else x_max
    x_domain = [x_min-.5, x_max+.5]

    # Create a series of values indexed by (y1 + each value of x)
    df = df.reset_index()[x+[y]].set_index(y).stack().reset_index().rename(columns={"level_1": y_name, 0: x_name})

    graph = (alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(x_name, scale=alt.Scale(domain=x_domain)),
            y=y_name,
            color=alt.Color(y_name, legend=None),
        ))
    
    return graph

def grafico_esplora_predizioni(df, x_min:int=None, x_max:int=None):

    df = df.rename(columns={"Fantavoto Media Reale Anno Successivo": "Media Reale",
                                "Fantavoto Media Predetta Anno Successivo": "Media Predetta",
                                "Differenza Reale Fantavoto Media Anno Successivo": "Reale",
                                "Differenza Predetta Fantavoto Media Anno Successivo": "Predetta"})

    selector_player = alt.selection_point(fields=['Nome'], on="mouseover", empty="none") 

    graph1 = (scatter(df,
        x="Media Reale",
        y="Media Predetta",
        tooltip=["Nome", "Media Reale", "Errore Predizione"],
        allinea_assi=True,
        diagonali=True)
        .add_params(selector_player)
        )

    graph2 = (bars_multiple_features(df,
        x=["Reale", "Predetta"],
        y="Nome",
        x_name="Differenza Anno Precedente (se disponibile)",
        y_name=" ",
        x_min=x_min,
        x_max=x_max)
        .transform_filter(selector_player)
        )
    graph2 = graph2.properties(height=100)
    
    return alt.vconcat(graph1, graph2).resolve_scale(color='independent')

def density_chart(x:str, group:str=None, df=None, graph_type:str="area", x_title:str=None, group_title:str=None, y_title:str="Density", legend:bool=True, color_domain:list=None, color_range:list=None, opacity:float=.3, filled:bool=True, strokeDash:list=[0]):

    chart = alt.Chart(df) if df is not None else alt.Chart()

    if graph_type == "area":
        chart = chart.mark_area(opacity=opacity, filled=filled, strokeDash=strokeDash)
    elif graph_type == "line":
        chart = chart.mark_line(opacity=opacity, filled=filled, strokeDash=strokeDash)
    else:
        raise Exception("graph_type should be either area or line")
    
    x_title = x if x_title is None else x_title
    if group is None:
        chart = chart.transform_density(x, as_=[x, y_title])
        chart = chart.encode(x=alt.X(x, title=x_title), y=y_title+":Q")
    else:
        chart = chart.transform_density(x, as_=[x, y_title], groupby=[group])
        group_title = group if group_title is None else group_title
        if legend:
            color = alt.Color(group, title=group_title) if color_domain is None or color_range is None else alt.Color(group, scale=alt.Scale(domain=color_domain, range=color_range), title=group_title)
        else:
            color = alt.Color(group, title=group_title, legend=None) if color_domain is None or color_range is None else alt.Color(group, scale=alt.Scale(domain=color_domain, range=color_range), title=group_title, legend=None)
        chart = chart.encode(x=alt.X(x, title=x_title), y=y_title+":Q", color=color)
    
    return chart

def grafico_densita_predizioni(df, by_group:bool=True, facet:bool=False):

    color_domain=["P", "D", "C", "A"]
    color_range=["#F8AB12", "#65C723", "#136AF6", "#F21C3C"]
    
    if by_group:

        df = df.rename(columns={"Ruolo":"Ruolo_", "Ruolo Anno Successivo":"Ruolo"})

        predicted = density_chart(
            x="Fantavoto Media Predetta Anno Successivo",
            group="Ruolo",
            x_title="Media Fantavoto",
            y_title="Densità",
            group_title="Predetta",
            color_domain=color_domain,
            color_range=color_range,
            graph_type="line",
            filled=False,
            opacity=1,
            legend=not facet
            )
        
        real = density_chart(
            x="Fantavoto Media Reale Anno Successivo",
            group="Ruolo",
            x_title="Media Fantavoto",
            y_title="Densità",
            group_title="Reale",
            color_domain=color_domain,
            color_range=color_range,
            opacity=.2,
            legend=not facet
            )

    else:

        predicted = density_chart(
            x="Fantavoto Media Predetta Anno Successivo",
            x_title="Media Fantavoto",
            y_title="Densità",
            graph_type="line",
            filled=False,
            opacity=1
            )
        
        real = density_chart(
            x="Fantavoto Media Reale Anno Successivo",
            x_title="Media Fantavoto",
            y_title="Densità",
            color_domain=color_domain,
            color_range=color_range,
            opacity=.2
            )
                
    chart = alt.layer(real, predicted, data=df.reset_index())

    if facet and by_group:
        return chart.properties(height=300).facet(facet=alt.Column("Ruolo", title=None), columns=2)
    else:
        return chart.resolve_scale(color='independent')
  
################################################################################

def main():
    st.set_page_config(layout="wide")
    st.sidebar.title("FantAnalytics")
    st.sidebar.image("https://cdn3.iconfinder.com/data/icons/education-science-vol-2-1/512/soccer_football_sports_game-512.png", width=150) #"https://cdn-icons-png.flaticon.com/512/3588/3588441.png"

    sx, dx = st.columns(2)
    with dx:
        stagione = st.selectbox("", ["2022-23", "2023-24"])
    with sx:
        st.title(f"Stagione {stagione}")

    if stagione == "2022-23":
        
        # Sidebar
        st.sidebar.header(":gear: Filtri")
        eta_min = int(df["Età"].min())
        eta_max = int(df["Età"].max())
        slider_eta = st.sidebar.slider("Età", min_value=eta_min, max_value=eta_max, value=[eta_min, eta_max])
        slider_presenze = st.sidebar.slider("Presenze Anno Precedente", min_value=0, max_value=38, value=[5,38])
        ruoli = ["P", "D", "C", "A"]
        checkbox_ruoli = st.sidebar.multiselect("Ruoli", options=ruoli, default=ruoli)
        name_search = (st.sidebar.text_input("Nome Giocatore")).title()
        
        # Filtra Dati
        subset = df[df["Ruolo Anno Successivo"].isin(checkbox_ruoli)]
        subset = subset[(subset["Età"] >= slider_eta[0]) & (subset["Età"] < slider_eta[1] + 1)]
        subset = subset[(subset["Presenze"] >= slider_presenze[0]) & (subset["Presenze"] < slider_presenze[1] + 1)]
        if name_search != "":
            subset = subset[subset["Nome"].str.startswith(name_search)]

        # Check in caso di subset vuoto
        if subset.shape[0] > 0:

            st.header("Esplora Predizioni Fantavoto")
            sx, dx = st.columns(2)
            with sx:
                st.altair_chart(grafico_esplora_predizioni(subset, x_min=-3, x_max=3), use_container_width=True)
            with dx:
                with st.expander("Dettagli", expanded=True):
                    st.info('''
                        ✅ Una previsione perfetta poggia sulla diagonale azzurra\n
                        - Un punto ↖️ rappresenta una previsione sovrastimata, uno ↘️ una previsione sottostimata\n
                        - Un punto ↗️ rappresenta una media Fantavoto alta, uno ↙️ una media bassa\n
                        🔍 Passa il mouse su un punto per scoprire il nome di quel giocatore e popolare il secondo grafico
                        ''')
            
            st.divider()
            sx, dx = st.columns(2)
            with sx:
                st.header("Densità Predizioni Fantavoto")
            with dx:
                with st.expander("Dettagli"):
                    st.info('''
                        📚 Compara la distribuzione Predetta (➖) a quella Reale (🟣)
                        ''')
                    dx1, dx2 = st.columns(2)
                    with dx1:
                        toggle_by_role = st.toggle("Densità per Ruolo", value=False)
                    with dx2:
                        if toggle_by_role:
                            toggle_facet = st.toggle("Separa Grafici per Ruolo", value=True)
                        else:
                            toggle_facet = False
            if subset.shape[0] > 0:
                st.altair_chart(grafico_densita_predizioni(subset, by_group=toggle_by_role, facet=toggle_facet), use_container_width=True)

            st.divider()
            with st.container():
                st.header("Performance Modello")
                sx, dx = st.columns(2)
                with sx:
                    st.subheader(":dart: Errore Assoluto")
                    sx1, sx2 = st.columns(2)
                    with sx1:
                        st.markdown("### Medio:")
                    with sx2:
                        mae = mean_absolute_error(subset["Fantavoto Media Reale Anno Successivo"], subset["Fantavoto Media Predetta Anno Successivo"])
                        st.markdown(f"### {round(mae, 2)}")
                    sx1, sx2 = st.columns(2)
                    with sx1:
                        st.markdown("### Medio %:")
                    with sx2:
                        mape = mean_absolute_percentage_error(subset["Fantavoto Media Reale Anno Successivo"], subset["Fantavoto Media Predetta Anno Successivo"])
                        st.markdown(f"### {round(mape*100, 2)}%")
                    sx1, sx2 = st.columns(2)
                    with sx1:
                        st.markdown("### Massimo:")
                    with sx2:
                        max_error = subset["Errore Predizione"].max()
                        st.markdown(f"### {round(max_error, 2)}")
                with dx:
                    st.subheader(":chart_with_upwards_trend: Accuratezza del Trend Previsto")
                    with st.expander("Dettagli"):
                        st.info('''
                                📉 Classificazione binaria: il modello riesce a predire in che direzione si muoverà la media Fantavoto rispetto all'anno precedente?\n
                                ❌ La seconda variante esclude le predizioni con errore trascurabile, per testare l'affidabilità del modello su quelle più problematiche\n
                                ⚙️ Usa lo slider per regolare la soglia di tolleranza\n
                                ⚠️ Questo KPI è ovviamente calcolato solo per i giocatori con Presenze Anno Precedente > 0
                                ''')
                        slider_soglia_errore = st.slider("Soglia Errore Trascurabile", min_value=0.0, max_value=1.0, value=0.5)
                    if slider_presenze[1] > 0:
                        dx1, dx2 = st.columns(2)
                        with dx1:
                            st.markdown(f"### Assoluta:")
                        with dx2:
                            acc_sign = accuracy_sign(subset["Fantavoto Media Reale Anno Successivo"], subset["Fantavoto Media Predetta Anno Successivo"], subset["Fantavoto Media"])
                            st.markdown(f"### {round(acc_sign*100, 2)}%")
                        dx1, dx2 = st.columns(2)
                        with dx1:
                            st.markdown(f"### Solo Predizioni con Errore >= {slider_soglia_errore}:")
                        with dx2:
                            acc_sign_given_error = accuracy_sign(subset["Fantavoto Media Reale Anno Successivo"], subset["Fantavoto Media Predetta Anno Successivo"], subset["Fantavoto Media"], given_threshold=slider_soglia_errore, absolute_error=True)
                            st.markdown(f"### {round(acc_sign_given_error*100, 2)}%")
                    else:
                        st.markdown("Non applicabile per giocatori senza Presenze Anno Precedente")
        
        else:
            st.divider()
            st.subheader("Nessun Giocatore :face_with_monocle:")
            st.markdown("Imposta dei filtri meno restrittivi")
            st.divider()
  
    elif stagione == "2023-24":
        st.markdown("## Coming soon...\n#### ...dopo le mie aste :sunglasses:\n######  \n###### (ma se non giocate contro di me scrivetemi pure)")

if __name__ == '__main__':
    main()
