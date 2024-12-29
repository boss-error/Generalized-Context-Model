from flask import Flask, render_template, request
import numpy as np
import pandas as pd

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        x_input = request.form.getlist('A')
        y_input = request.form.getlist('B')
        
        x_tuples = [tuple(map(int, item.split(','))) for item in x_input]
        y_tuples = [tuple(map(int, item.split(','))) for item in y_input]

        predict_observation = request.form['observation']
        observation = np.array([int(i) for i in predict_observation.split(',')])

        # Example data and observation
        C = float(request.form['C_value'])
        # Data for A and B
        data = pd.DataFrame({
            'A': x_tuples,
            'B': y_tuples
        })

        # Calculate distances d(A) and d(B)
        data['d(A)'] = data['A'].apply(lambda x: np.sqrt(np.sum((np.array(x) - observation) ** 2)))
        data['d(B)'] = data['B'].apply(lambda x: np.sqrt(np.sum((np.array(x) - observation) ** 2)))

        # Similarity calculation
        data['Si,j (A)'] = data['A'].apply(lambda x: np.exp(-C * np.sqrt(np.sum((np.array(x) - observation) ** 2))))
        data['Si,j (B)'] = data['B'].apply(lambda x: np.exp(-C * np.sqrt(np.sum((np.array(x) - observation) ** 2))))

        # Summing similarity values
        sum_Si_A = data['Si,j (A)'].sum()
        sum_Si_B = data['Si,j (B)'].sum()

        # Probabilities
        P_A = sum_Si_A / (sum_Si_A + sum_Si_B)
        P_B = sum_Si_B / (sum_Si_A + sum_Si_B)

        # Classification based on probabilities
        classification = "Class A" if P_A > P_B else "Class B"

        # Generate LaTeX outputs
        table_latex = generate_latex_table(data, sum_Si_A, sum_Si_B, P_A, P_B, classification)
        equations_latex = generate_equations(C, observation,data)
        steps_latex = generate_steps(sum_Si_A, sum_Si_B, P_A, P_B,data, observation,C)
        result_latex = f"Classification: {classification}"

        return render_template('solver.html', 
                               table_latex=table_latex, 
                               equations_latex=equations_latex,
                               steps_latex=steps_latex,
                               result_latex=result_latex)

    return render_template('index.html')

def generate_latex_table(data, sum_Si_A, sum_Si_B, P_A, P_B, classification):
    latex_table = f"""
    \\begin{{array}}{{|c|c|c|c|c|}} \\hline
    A & B & d(A) & d(B) & S_{{i,j}}(A) & S_{{i,j}}(B) \\\\ \\hline
    """

    for index, row in data.iterrows():
        latex_table += f"{row['A']} & {row['B']} & {row['d(A)']:.4f} & {row['d(B)']:.4f} & {row['Si,j (A)']:.4f} & {row['Si,j (B)']:.4f} \\\\ \\hline\n"

    latex_table += f"""
    \\text{{Σ S}}_{{i,j}}(A) & & & & {sum_Si_A:.4f} & {sum_Si_B:.4f} \\\\ \\hline
    \\text{{P(R1=A/i)}} & & & & {P_A:.4f} & \\\\ \\hline
    \\text{{P(R1=B/i)}} & & & & {P_B:.4f} & \\\\ \\hline
    \\text{{Classification}} & & & & {classification} & \\\\ \\hline
    \\end{{array}}
    """
    return latex_table
def generate_steps(sum_Si_A, sum_Si_B, P_A, P_B, data, observation, C):
    steps = f"""
    <ul>
        <li>Substituting for d(A) and d(B):</li>
        <ul>
    """
    for index, row in data.iterrows():
        # Dynamic calculation for d(A) and d(B)
        d_A_elements = " + ".join([f"({row['A'][i]} - {observation[i]})^2" for i in range(len(row['A']))])
        d_B_elements = " + ".join([f"({row['B'][i]} - {observation[i]})^2" for i in range(len(row['B']))])
        steps += f"""
            <li>
                \\( d(A) = \\sqrt{{{d_A_elements}}} = {row['d(A)']:.4f} \\)
            </li>
            <li>
                \\( d(B) = \\sqrt{{{d_B_elements}}} = {row['d(B)']:.4f} \\)
            </li>
        """
    steps += f"""
        </ul>
        <li>Substituting for S(i,j)(A) and S(i,j)(B):</li>
        <ul>
    """
    for index, row in data.iterrows():
        steps += f"""
            <li>
                \\( S_{{i,j}}(A) = e^{{-{C} \\cdot {row['d(A)']:.4f}}} = {row['Si,j (A)']:.4f} \\)
            </li>
            <li>
                \\( S_{{i,j}}(B) = e^{{-{C} \\cdot {row['d(B)']:.4f}}} = {row['Si,j (B)']:.4f} \\)
            </li>
        """
    steps += f"""
        </ul>
        <li>Substituting for P(A) and P(B):</li>
        <ul>
            <li>
                \\( P(A) = \\frac{{{sum_Si_A:.4f}}}{{{sum_Si_A:.4f} + {sum_Si_B:.4f}}} = {P_A:.4f} \\)
            </li>
            <li>
                \\( P(B) = \\frac{{{sum_Si_B:.4f}}}{{{sum_Si_A:.4f} + {sum_Si_B:.4f}}} = {P_B:.4f} \\)
            </li>
        </ul>
    </ul>
    """
    return steps





def generate_equations(C, observation, data):
    equations = f"""
    $$\\mathbf{{Equations\\ for\\ d(A)\\ and\\ d(B)}}:$$
    """
    for index, row in data.iterrows():
        d_A_elements = " + ".join([f"(A{i} - C)^2" for i in range(len(row['A']))])

        d_B_elements = " + ".join([f"(B{i} - C)^2" for i in range(len(row['B']))])
        equations += f"""
        $$d(A) = \\sqrt{{{d_A_elements}}}$$
        $$d(B) = \\sqrt{{{d_B_elements}}}$$
        """
    
    equations += f"""
    $$\\mathbf{{Similarity\\ Formula:}}$$
    $$S_{{i,j}}(A) = e^{{-C \\cdot d(A)}}$$
    $$S_{{i,j}}(B) = e^{{-C \\cdot d(B)}}$$

    $$\\mathbf{{Probability\\ Formula:}}$$
    $$\\text{{Equation for P(A):}} \\quad P(A) = \\frac{{\\Sigma S_{{i,j}}(A)}}{{\\Sigma S_{{i,j}}(A) + \\Sigma S_{{i,j}}(B)}}$$
    $$\\text{{Equation for P(B):}} \\quad P(B) = \\frac{{\\Sigma S_{{i,j}}(B)}}{{\\Sigma S_{{i,j}}(A) + \\Sigma S_{{i,j}}(B)}}$$
    """
    return equations









if __name__ == '__main__':
    app.run(debug=True)
