#import necessary libraries 
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk

#Data Loading and Preprocessing

df = pd.read_csv("realtor-data.csv")
for_sale_df = df[df["status"] == "for_sale"].copy()
ready_to_build_df = df[df["status"] == "ready_to_build"].copy()

# we convert comma to point for python (original table uses comma)

for_sale_df['acre_lot'] = for_sale_df['acre_lot'].astype(str).str.replace(',', '.').astype(float)
ready_to_build_df['acre_lot'] = ready_to_build_df['acre_lot'].astype(str).str.replace(',', '.').astype(float)

#check empty columns 
for_sale_df = for_sale_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
ready_to_build_df = ready_to_build_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)


#to show in treeview original names
city_mapping_for_sale = dict(enumerate(for_sale_df['city'].astype('category').cat.categories))
state_mapping_for_sale = dict(enumerate(for_sale_df['state'].astype('category').cat.categories))

city_mapping_ready = dict(enumerate(ready_to_build_df['city'].astype('category').cat.categories))
state_mapping_ready = dict(enumerate(ready_to_build_df['state'].astype('category').cat.categories))
combined_city_mapping = {**city_mapping_for_sale, **city_mapping_ready}
combined_state_mapping = {**state_mapping_for_sale, **state_mapping_ready}



#labeling: 
for_sale_df['city'] = for_sale_df['city'].astype('category').cat.codes
for_sale_df['state'] = for_sale_df['state'].astype('category').cat.codes
ready_to_build_df['city'] = ready_to_build_df['city'].astype('category').cat.codes
ready_to_build_df['state'] = ready_to_build_df['state'].astype('category').cat.codes

#some columns should be omitted 
columns_to_drop = ['full_address', 'sold_date', 'street']
for_sale_df.drop(columns=columns_to_drop, inplace=True)
ready_to_build_df.drop(columns=columns_to_drop, inplace=True)

combined_df = pd.concat([for_sale_df, ready_to_build_df], axis=0)
combined_df = pd.get_dummies(combined_df, columns=['status'])
combined_df.fillna(combined_df.mean(), inplace=True)

# split as X and y

X = combined_df.drop("price", axis=1)  
y = combined_df["price"]

# Splitting and Model Training]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#regression models 
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

lasso_model = Lasso(alpha=1.0)
lasso_model.fit(X_train, y_train)

#train a decision tree regressor
tree_model = DecisionTreeRegressor(max_depth=4) 
tree_model.fit(X_train, y_train)

# Predict on the test set
y_pred_tree = tree_model.predict(X_test)

# Evaluate the Decision Tree
mae_tree = mean_absolute_error(y_test, y_pred_tree)
mse_tree = mean_squared_error(y_test, y_pred_tree)
r2_tree = r2_score(y_test, y_pred_tree)

print(f"Decision Tree - MAE: {mae_tree}")
print(f"Decision Tree - MSE: {mse_tree}")
print(f"Decision Tree - R2: {r2_tree}")

# Graphics for Decision Tree
fig_tree, ax_tree = plt.subplots(figsize=(15, 15),dpi=300)

plot_tree(tree_model, feature_names=X.columns.tolist(), filled=True, rounded=True, ax=ax_tree)


#display tree 
def display_tree():
    tree_window = tk.Toplevel(root)
    tree_window.title("Decision Tree Visualization")
    
    # Adjust the figure size
    fig_tree.set_size_inches(10, 10)  # This sets it to 10x10 inches. Adjust as needed.
    
    canvas_tree = FigureCanvasTkAgg(fig_tree, master=tree_window)
    canvas_tree.draw()
    canvas_tree.get_tk_widget().pack(fill=tk.BOTH, expand=True)  # Allow canvas to expand
    
    #tree window size 
    tree_window.geometry('1000x1000')  # Set to 1000x1000 pixels



#predict price 

def predict_price():
    try:
        model_name = model_spinbox.get()
        model_dict = {
            "Linear Regression": linear_model,
            "Ridge Regression": ridge_model,
            "Lasso Regression": lasso_model,
            "Decision Tree": tree_model
        }
        model_selected = model_dict[model_name]
        
        # Get the input values  fields
        input_values = {}
        for feature in features:  
            input_values[feature] = float(entries[feature].get())

        input_values_list = [input_values[feature] for feature in features]
        prediction = model_selected.predict([input_values_list])[0]
        
        # Display prediction in the textbox
        prediction_text.set(f"Predicted Price: ${prediction:.2f}\nUsing Model: {model_name}")
        
        print(f"Predicted Price: ${prediction:.2f}\nUsing Model: {model_name}") #test 
        prediction_display.config(text=f"Predicted Price: ${prediction:.2f}\nUsing Model: {model_name}")
    except Exception as e:
        prediction_text.set(f"Error: {str(e)}")
        print(f"Error: {str(e)}")  # Print error for debugging


def display_parameter_entry():
    global code_display
    # to create and display the child window for parameters entry and conversion.
    
    # Child window
    param_window = tk.Toplevel(root)
    param_window.title("Enter Parameters")

    # Label and entry for city
    city_label = ttk.Label(param_window, text="City:")
    city_label.grid(row=0, column=0, padx=5, pady=5)

    city_entry = ttk.Entry(param_window)
    city_entry.grid(row=0, column=1, padx=5, pady=5)

    # Label and entry for state
    state_label = ttk.Label(param_window, text="State:")
    state_label.grid(row=1, column=0, padx=5, pady=5)

    state_entry = ttk.Entry(param_window)
    state_entry.grid(row=1, column=1, padx=5, pady=5)

    # Button to convert inputs to codes
    convert_button = ttk.Button(param_window, text="Convert to Codes", command=lambda: convert_to_codes(city_entry, state_entry))
    convert_button.grid(row=2, column=0, columnspan=2, pady=10)

    # Text box to display coded values
    code_display = tk.Text(param_window, height=5, width=30)
    code_display.grid(row=3, column=0, columnspan=2, padx=5, pady=5)


    def convert_to_codes(city_entry, state_entry):
    
    # Fetch values from entry fields 
        city_val = city_entry.get()
        state_val = state_entry.get()

    # Convert city name and state name to its code
        city_code = next((code for code, name in combined_city_mapping.items() if name == city_val), None)
        state_code = next((code for code, name in combined_state_mapping.items() if name == state_val), None)

    # Display in the text box
        code_display.delete(1.0, tk.END)  # Clear existing content
        code_display.insert(tk.END, f"City Code: {city_code}\nState Code: {state_code}")



# GUI setup
root = tk.Tk()
root.title("Regression Analysis with Different Models")
root.config(bg="lightgray")

frame1 = ttk.LabelFrame(root, text="Metrics", padding="5")
frame1.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

models = [linear_model, ridge_model, lasso_model,tree_model]
names = ["Linear Regression", "Ridge Regression", "Lasso Regression","Decision Tree"]
predictions = [model.predict(X_test) for model in models]

for index, (model, name) in enumerate(zip(models, names)):
    y_pred = model.predict(X_test)
    
    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics_text = f"{name}\nMAE: {mae}\nMSE: {mse}\nR2: {r2}"
    metrics_label = ttk.Label(frame1, text=metrics_text, relief="solid", padding=(10, 10))
    metrics_label.grid(row=index, column=0, sticky="w", pady=5, padx=5)

    # Actual vs Predicted Plot for Regression models
    if name != "Decision Tree":
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title(f"Actual vs. Predicted for {name}")
        ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
        
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().grid(row=(index % 2), column=(index // 2) + 1, padx=10, pady=10)

# Button to display the Decision Tree
tree_button = ttk.Button(root, text="Display Decision Tree", command=display_tree)
tree_button.place(x=10,y=500)  



# Frame for Predictions
frame2 = ttk.LabelFrame(root, text="Predict House Price", padding="5")
frame2.grid(row=0, column=3, rowspan=4, padx=10, pady=10, sticky="nsew")

# Input fields for each feature
features = list(X.columns)
entries = {}

for idx, feature in enumerate(features):
    label = ttk.Label(frame2, text=f"Enter {feature}:")
    label.grid(row=idx, column=0, padx=5, pady=5)
    
    entry = ttk.Entry(frame2, width=30)
    entry.grid(row=idx, column=1, padx=5, pady=5)
    
    entries[feature] = entry

# Spinbox for Model Selection
model_label = ttk.Label(frame2, text="Select Model:")
model_label.grid(row=len(features), column=0, padx=5, pady=5)

model_names = ["Linear Regression", "Ridge Regression", "Lasso Regression", "Decision Tree"]
model_spinbox = ttk.Spinbox(frame2, values=model_names, width=20)
model_spinbox.grid(row=len(features), column=1, padx=5, pady=5)




prediction_text = tk.StringVar()
prediction_display=tk.Label(frame2,width=30,borderwidth=3,bg="powderblue") #root
prediction_display.place(x=50,y=400)



predict_button = ttk.Button(frame2, text="Predict Price",command=predict_price,width=20)

predict_button.place(x=50,y=440)



# Frame for Sample Data
frame3 = ttk.LabelFrame(root, text="Sample Data", padding="5",relief="sunken")
frame3.grid(row=2, column=1, padx=10, pady=10, sticky="nsew")  

# Create the treeview
tree = ttk.Treeview(frame3, columns=features, show="headings")

# Define the column headings
for feature in features:
    tree.heading(feature, text=feature)
    tree.column(feature, width=100, anchor='center')  # Adjust the width for better visibility

# Insert sample data into the treeview
sample_data = combined_df.head(10)

#try to put original names

for index, row in sample_data.iterrows():
    values = []
    for feature in features:
        if feature == 'city':
            values.append(combined_city_mapping[row[feature]])
        elif feature == 'state':
            values.append(combined_state_mapping[row[feature]])
        else:
            values.append(row[feature])
    tree.insert("", "end", values=tuple(values))


#  treeview place
tree.pack(fill="both", expand=True)

# Scrollbar for treeview
scroll = ttk.Scrollbar(frame3, orient="vertical", command=tree.yview)
scroll.pack(side='right', fill='y')
tree.configure(yscrollcommand=scroll.set)



# Add File menu
menu_bar = tk.Menu(root)
root.config(menu=menu_bar)

file_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="File", menu=file_menu)

file_menu.add_command(label="Parameters Entry", command=display_parameter_entry)
file_menu.add_command(label="Exit", command=root.quit)



root.mainloop()

#end program 
