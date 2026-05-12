from flask import Flask, render_template, request
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import pickle
import sklearn

app = Flask(__name__)

print("Flask using sklearn version:", sklearn.__version__)

# Load trained pipeline
with open("Paddyield.pkl", "rb") as f:
    model = pickle.load(f)
print("Loaded model:", model)

@app.route("/", methods=["GET", "POST"])
def home():
    predict = None
    error_msg = None
    
    if request.method == "POST":
        try:
            #Hectares_                      = float(request.form["Hectares_"])
            Agriblock                      = float(request.form["Agriblock"])
            Variety                        = float(request.form["Variety"])
            Soil_Types                     = float(request.form["Soil_Types"])
            #Seedrate_in_Kg                 = float(request.form["Seedratein_Kg"])
            #LP_Mainfield_in_Tonnes         = float(request.form["LP_Mainfieldin_Tonnes"])
            Nursery                        = float(request.form["Nursery"])
            #Nursery_area_Cents             = float(request.form["Nursery_area_Cents"])
            LP_nurseryarea_in_Tonnes       = float(request.form["LP_nurseryareain_Tonnes"])
            #DAP_20days                     = float(request.form["DAP_20days"])
            Weed28D_thiobencarb            = float(request.form["Weed28D_thiobencarb"])
            #Urea_40Days                    = float(request.form["Urea_40Days"])
            #Potassh_50Days                 = float(request.form["Potassh_50Days"])
            #Micronutrients_70Days          = float(request.form["Micronutrients_70Days"])
            Pest_60Day_in_ml               = float(request.form["Pest_60Dayin_ml"])
            #Rain30_in_mm                   = float(request.form["30DRain_in_mm"])
            #AI30_in_mm                     = float(request.form["30DAIin_mm"])
            #Rain30_50_in_mm                = float(request.form["30_50DRain_in_mm"])
            #AI30_50_in_mm                  = float(request.form["30_50DAIin_mm"])
            Rain51_70_in_mm                = float(request.form["51_70DRainin_mm"])
            AI51_70_in_mm                  = float(request.form["51_70AIin_mm"])
            #Rain71_105_in_mm               = float(request.form["71_105DRainin_mm"])
            AI71_105_in_mm                 = float(request.form["71_105DAIin_mm"])
            #Min_temp_D1_D30                = float(request.form["Min_temp_D1_D30"])
            Max_temp_D1_D30                = float(request.form["Max_temp_D1_D30"])
            Min_temp_D31_D60               = float(request.form["Min_temp_D31_D60"])
            #Max_temp_D31_D60               = float(request.form["Max_temp_D31_D60"])
            Min_temp_D61_D90               = float(request.form["Min_temp_D61_D90"])
            Max_temp_D61_D90               = float(request.form["Max_temp_D61_D90"])
            Min_temp_D91_D120              = float(request.form["Min_temp_D91_D120"])
            #Max_temp_D91_D120              = float(request.form["Max_temp_D91_D120"])
            #Inst_Wind_Speed_D1_D30_in_Knots  = float(request.form["Inst_Wind_Speed_D1_D30in_Knots"])
            Inst_Wind_Speed_D31_D60_in_Knots = float(request.form["Inst_Wind_Speed_D31_D60in_Knots"])
            #Inst_Wind_Speed_D61_D90_in_Knots = float(request.form["Inst_Wind_Speed_D61_D90in_Knots"])
            Inst_Wind_Speed_D91_D120_in_Knots= float(request.form["Inst_Wind_Speed_D91_D120in_Knots"])
            #Wind_Direction_D1_D30          = float(request.form["Wind_Direction_D1_D30"])
            #Wind_Direction_D31_D60         = float(request.form["Wind_Direction_D31_D60"])
            Wind_Direction_D61_D90         = float(request.form["Wind_Direction_D61_D90"])
            Wind_Direction_D91_D120        = float(request.form["Wind_Direction_D91_D120"])
            Relative_Humidity_D1_D30       = float(request.form["Relative_Humidity_D1_D30"])
            Relative_Humidity_D31_D60      = float(request.form["Relative_Humidity_D31_D60"])
            #Relative_Humidity_D61_D90      = float(request.form["Relative_Humidity_D61_D90"])
            Relative_Humidity_D91_D120     = float(request.form["Relative_Humidity_D91_D120"])   # <- matches HTML name
            
            # Build DataFrame with EXACT training column names
            #input_df = pd.DataFrame([{
            #    "total_sqft": total_sqft,
            #    "bath": bath,
            #    "BHK": BHK,
            #    "balcony": balcony,
            #    "area_type": area_type
            #}])

            sample_dict = {
                        #"Hectares ": Hectares_,
                        "Agriblock": Agriblock,
                        "Variety": Variety,
                        "Soil Types": Soil_Types,
                        #"Seedrate(in Kg)": Seedrate_in_Kg,
                        #"LP_Mainfield(in Tonnes)": LP_Mainfield_in_Tonnes,
                        "Nursery": Nursery,
                        #"Nursery area (Cents)": Nursery_area_Cents,
                        "LP_nurseryarea(in Tonnes)": LP_nurseryarea_in_Tonnes,
                        #"DAP_20days": DAP_20days,
                        "Weed28D_thiobencarb": Weed28D_thiobencarb,
                        #"Urea_40Days": Urea_40Days,
                        #"Potassh_50Days": Potassh_50Days,
                        #"Micronutrients_70Days": Micronutrients_70Days,
                        "Pest_60Day(in ml)": Pest_60Day_in_ml,
                        #"30DRain( in mm)": Rain30_in_mm,
                        #"30DAI(in mm)": AI30_in_mm,
                        #"30_50DRain( in mm)": Rain30_50_in_mm,
                        #"30_50DAI(in mm)": AI30_50_in_mm,
                        "51_70DRain(in mm)": Rain51_70_in_mm,
                        "51_70AI(in mm)": AI51_70_in_mm,
                        #"71_105DRain(in mm)": Rain71_105_in_mm,
                        "71_105DAI(in mm)": AI71_105_in_mm,
                        #"Min temp_D1_D30": Min_temp_D1_D30,
                        "Max temp_D1_D30": Max_temp_D1_D30,
                        "Min temp_D31_D60": Min_temp_D31_D60,
                        #"Max temp_D31_D60": Max_temp_D31_D60,
                        "Min temp_D61_D90": Min_temp_D61_D90,
                        "Max temp_D61_D90": Max_temp_D61_D90,
                        "Min temp_D91_D120": Min_temp_D91_D120,
                        #"Max temp_D91_D120": Max_temp_D91_D120,
                        #"Inst Wind Speed_D1_D30(in Knots)": Inst_Wind_Speed_D1_D30_in_Knots,
                        "Inst Wind Speed_D31_D60(in Knots)": Inst_Wind_Speed_D31_D60_in_Knots,
                        #"Inst Wind Speed_D61_D90(in Knots)": Inst_Wind_Speed_D61_D90_in_Knots,
                        "Inst Wind Speed_D91_D120(in Knots)": Inst_Wind_Speed_D91_D120_in_Knots,
                        #"Wind Direction_D1_D30": Wind_Direction_D1_D30,
                        #"Wind Direction_D31_D60": Wind_Direction_D31_D60,
                        "Wind Direction_D61_D90": Wind_Direction_D61_D90,
                        "Wind Direction_D91_D120": Wind_Direction_D91_D120,
                        "Relative Humidity_D1_D30": Relative_Humidity_D1_D30,
                        "Relative Humidity_D31_D60": Relative_Humidity_D31_D60,
                        #"Relative Humidity_D61_D90": Relative_Humidity_D61_D90,
                        "Relative Humidity_D91_D120": Relative_Humidity_D91_D120}

            input_df = pd.DataFrame([sample_dict])

            # Use the pipeline the way it was trained
            pred = model.predict(input_df)[0]
            predict = round(float(pred), 2)


        except Exception as e:
            error_msg = repr(e)
            print("Error during prediction:", error_msg)
            predict = "Invalid Input"

            #print("DEBUG value:", sample_dict)
            #print("DEBUG type:", type(sample_dict))

    # Pass error to template too so you can see it while debugging
    return render_template("index.html", predict=predict, error=error_msg)

if __name__ == "__main__":
    app.run(debug=True)