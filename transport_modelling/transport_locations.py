# %%
import pandas as pd
import googlemaps
import json
import numpy as np

# %%
# Metro list - obtained using ATM maps and Wikipedia list of metro stations
# ATM - https://www.atm.it/en/ViaggiaConNoi/Pages/SchemaReteMetro.aspx
# Wikipedia - https://en.wikipedia.org/wiki/List_of_Milan_Metro_stations
metro = ["Abbiategrasso (Milan Metro)"
            , "Affori Centro (Milan Metro)"
            , "Affori FN (Milan Metro)"
            , "Amendola (Milan Metro)"
            , "Assago Milanofiori Forum (Milan Metro)"
            , "Assago Milanofiori Nord (Milan Metro)"
            , "Bande Nere (Milan Metro)"
            , "Bicocca (Milan Metro)"
            , "Bignami (Milan Metro)"
            , "Bisceglie (Milan Metro)"
            , "Bonola (Milan Metro)"
            , "Brenta (Milan Metro)"
            , "Buonarroti (Milan Metro)"
            , "Bussero (Milan Metro)"
            , "Ca’ Granda (Milan Metro)"
            , "Cadorna (Milan Metro)"
            , "Caiazzo (Milan Metro)"
            , "Cairoli (Milan Metro)"
            , "Cascina Antonietta (Milan Metro)"
            , "Cascina BurronaCa’ Granda (Milan Metro)"
            , "Cascina GobbaCa’ Granda (Milan Metro)"
            , "Cassina de' PecchiCa’ Granda (Milan Metro)"
            , "CenisioCa’ Granda (Milan Metro)"
            , "CentraleCa’ Granda (Milan Metro)"
            , "Cernusco sul Naviglio (Milan Metro)"
            , "Cimiano (Milan Metro)"
            , "Cologno Centro (Milan Metro)"
            , "Cologno Nord (Milan Metro)"
            , "Cologno Sud (Milan Metro)"
            , "Comasina (Milan Metro)"
            , "Conciliazione (Milan Metro)"
            , "Cordusio (Milan Metro)"
            , "Corvetto (Milan Metro)"
            , "Crescenzago (Milan Metro)"
            , "Crocetta (Milan Metro)"
            , "De Angeli (Milan Metro)"
            , "Dergano (Milan Metro)"
            , "Domodossola (Milan Metro)"
            , "Duomo (Milan Metro)"
            , "Famagosta (Milan Metro)"
            , "Gambara (Milan Metro)"
            , "Porta Garibaldi (Milan Metro)"
            , "Gerusalemme (Milan Metro)"
            , "Gessate (Milan Metro)"
            , "Gioia (Milan Metro)"
            , "Gorgonzola (Milan Metro)"
            , "Gorla (Milan Metro)"
            , "Inganni (Milan Metro)"
            , "Isola (Milan Metro)"
            , "Istria (Milan Metro)"
            , "Lambrate (Milan Metro)"
            , "Lampugnano (Milan Metro)"
            , "Lanza (Milan Metro)"
            , "Lima (Milan Metro)"
            , "Lodi T.I.B.B. (Milan Metro)"
            , "Loreto (Milan Metro)"
            , "Lotto (Milan Metro)"
            , "Maciachini (Milan Metro)"
            , "Marche (Milan Metro)"
            , "Missori (Milan Metro)"
            , "Molino Dorino (Milan Metro)"
            , "Montenapoleone (Milan Metro)"
            , "Monumentale (Milan Metro)"
            , "Moscova (Milan Metro)"
            , "Pagano (Milan Metro)"
            , "Palestro (Milan Metro)"
            , "Pasteur (Milan Metro)"
            , "Pero (Milan Metro)"
            , "Piola (Milan Metro)"
            , "Ponale (Milan Metro)"
            , "Porta Genova (Milan Metro)"
            , "Porta Romana (Milan Metro)"
            , "Porta Venezia (Milan Metro)"
            , "Portello (Milan Metro)"
            , "Porto di Mare (Milan Metro)"
            , "Precotto (Milan Metro)"
            , "Primaticcio (Milan Metro)"
            , "QT8 (Milan Metro)"
            , "Repubblica (Milan Metro)"
            , "Rho Fiera (Milan Metro)"
            , "Rogoredo (Milan Metro)"
            , "Romolo (Milan Metro)"
            , "Rovereto (Milan Metro)"
            , "San Babila (Milan Metro)"
            , "San Donato (Milan Metro)"
            , "San Leonardo (Milan Metro)"
            , "San Siro Ippodromo (Milan Metro)"
            , "San Siro Stadio (Milan Metro)"
            , "Sant'Agostino (Milan Metro)"
            , "Sant'Ambrogio (Milan Metro)"
            , "Segesta (Milan Metro)"
            , "Sesto 1º Maggio (Milan Metro)"
            , "Sesto Marelli (Milan Metro)"
            , "Sesto Rondò (Milan Metro)"
            , "Sondrio (Milan Metro)"
            , "Tre Torri (Milan Metro)"
            , "Turati (Milan Metro)"
            , "Turro (Milan Metro)"
            , "Udine (Milan Metro)"
            , "Uruguay (Milan Metro)"
            , "Villa Fiorita (Milan Metro)"
            , "Villa Pompea (Milan Metro)"
            , "Villa San Giovanni (Milan Metro)"
            , "Vimodrone (Milan Metro)"
            , "Wagner (Milan Metro)"
            , "Zara (Milan Metro)"]

# Bus list - Obtained from Google Maps
bus = ['Terravision bus stop, Milan', 'Piazza Iv Novembre Stazione Centrale, Milan'
        , 'Via M.te Pietà prima di Via Croce Rossa, Milan', 'Via F.Sforza 48 prima di C.so di P.ta Romana, Milan'
        , 'Via Boccaccio prima di Via A.Saffi, Milan', 'Via Donizetti 48 prima di C.so Monforte, Milan'
        , 'P.za S.Maria Nascente, Milan', 'Via Noale dopo Via Val Badia, Milan', ' P.za M.te Falterona 5, Milan'
        , 'Via M.te S.Gabriele dopo V.le Monza, Milan', 'Via De Marchi dopo Via Montebello, Milan'
        , 'Fermata Movibus, Milan', ' P.za della Repubblica Lato 14 dopo V.le Vittorio Veneto, Milan'
        , 'V.le Monza 208 dopo Via Don Guanella, Milan', 'Via Turati prima di Via Mangili, Milan'
        , 'Via Molino delle Armi prima di C.so di P.ta Ticinese, Milan', 'Via Carducci 13 prima di C.so Magenta, Milan'
        , 'Via E.De Marchi Altezza P.za Greco, Milan', 'V.le Romagna 69 dopo P.le Piola, Milan'
        , 'Loreto M1 m2, Milan']

# Tram list - Obtained from Google Maps
tram = ['Via Martiri Oscuri Fronte 21, Milan', 'Via S.Margherita prima di P.za della Scala, Milan', 'P.za Fontana, Milan'
        , 'V.le Piave 27 prima di Via Morelli, Milan', 'Via del Turchino 13 prima di Via Maspero, Milan'
        , 'V.le Lunigiana dopo Via M.Gioia, Milan', 'Via V.Monti prima di P.za Giovanni Xxiii., Milan'
        , 'Cadorna FN M1 M2, Milan', 'L.go Donegani 2 prima di Via Moscova, Milan', 'Duomo, Milan'
        , 'Via C.Marcello prima di Via Grugnola, Milan', 'C.so Sempione Fronte 77 prima di Via Em.Filiberto, Milan'
        , 'C.so Sempione 27 prima di Via Villasanta, Milan', 'Stazione Garibaldi, Milan', 'Via Schiaparelli prima di Via Ponte Seveso, Milan'
        , 'P.za L. Da Vinci Fronte 9 prima di Via Spinoza, Milan', 'Via degli Scipioni 1/A prima di P.za M.Adelaide di Savoia, Milan'
        , 'Duomo M1 M3, Milan', 'Via B.Angelico 17 prima di Via Colombo, Milan', 'P.za 6 Febbraio 4 prima di Via Vincenzo Monti, Milan']

# %%
# Initializing API
gmaps = googlemaps.Client(key='?')

locations_final = np.NaN

# %%
# Metro requests
for location in metro:
    geocode_result = gmaps.geocode(location)

    locations = pd.DataFrame([s["geometry"]['location'] for s in geocode_result]) # Get lat long position

    locations['loc'] = location
    locations['type'] = 'metro'

    if type(locations_final) == type(np.NaN):
        locations_final = locations 
    else:
        locations_final = locations_final.append(locations)

# %%
# Bus requests
for location in bus:
    geocode_result = gmaps.geocode(location)

    locations = pd.DataFrame([s["geometry"]['location'] for s in geocode_result]) # Get lat long position

    locations['loc'] = location
    locations['type'] = 'bus'

    if type(locations_final) == type(np.NaN):
        locations_final = locations 
    else:
        locations_final = locations_final.append(locations)
        
# %%
# Tram requests
for location in tram:
    geocode_result = gmaps.geocode(location)

    locations = pd.DataFrame([s["geometry"]['location'] for s in geocode_result]) # Get lat long position

    locations['loc'] = location
    locations['type'] = 'tram'

    if type(locations_final) == type(np.NaN):
        locations_final = locations 
    else:
        locations_final = locations_final.append(locations)

# %%
locations_final['loc'] = locations_final['loc'].apply(lambda x: x.replace(" (Milan Metro)", ""))
locations_final['loc'] = locations_final['loc'].apply(lambda x: x.replace(", Milan", ""))

locations_final = locations_final.reset_index().drop(['index'], axis=1)

# %%
locations_final.to_csv('public_transport_locations.csv')
# %%
