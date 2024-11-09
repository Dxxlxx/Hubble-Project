import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # ใช้ Agg backend สำหรับบันทึกกราฟเป็นไฟล์แทนการแสดงผลบนหน้าจอ
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

# ข้อมูล JSON ที่มี
galaxy_data = {
    "Apparent Magitude (m)": [14.541391237647325, 16.756128786151876, 15.829604890719585, 16.29457480995023, 13.829093893418786, 14.182568722430029, 16.03907734592769, 17.138868011663146, 15.998250803312972, 17.910486867901746, 15.074669722972802, 15.164039588647865, 12.484639187522449, 15.60165860026537, 13.753617662388354, 14.50453086130131, 14.147029911292748, 16.04484713334022, 15.192141049675008],
    "Absolute Magnitude (M)": [-16.339065057631082, -18.177729885179346, -16.696145000879945, -19.512265201224647, -19.665756128261307, -21.24923043094372, -17.778062621886992, -17.656338949942327, -19.397655426925155, -16.93192787486793, -20.918615711799312, -18.86686028127157, -23.32220082365243, -18.32499057478847, -21.310568461137503, -18.80925829710656, -18.51036467391853, -20.29101150867485, -16.313008928644898],
    "Redshift (z)": [0.0036759669738346545, 0.02260424055745114, 0.008037853020448127, 0.036351715342596735, 0.01039452291356091, 0.03358661872848456, 0.014447634338827342, 0.017837996244636223, 0.030485281173716405, 0.025338258561158566, 0.030141298549296147, 0.017837996244636223, 0.0325517381815954, 0.011068888391776666, 0.01715898899234225, 0.01208129940826752, 0.006022296587493203, 0.03220702315577939, 0.004010818525698623]
}

# แปลงข้อมูล JSON เป็น DataFrame ของ Pandas
galaxy_df = pd.DataFrame(galaxy_data)

# ฟังก์ชันคำนวณระยะทาง (d) และความเร็ว (v) ของกาแล็กซี่
def calculate_distance_velocity(galaxy_df):
    c = 3 * 10**5  # ความเร็วแสง km/s
    
    # ดึงข้อมูลจาก DataFrame
    m_values = galaxy_df["Apparent Magitude (m)"]
    M_values = galaxy_df["Absolute Magnitude (M)"]
    z_values = galaxy_df["Redshift (z)"]
    
    # คำนวณระยะทาง d
    distances = 10 ** ((m_values - M_values) / 5 + 1)  # หน่วยเป็น parsecs
    
    # คำนวณความเร็ว v
    velocities = z_values * c  # หน่วยเป็น km/s
    
    return distances.values, velocities.values

# คำนวณระยะทางและความเร็ว
distances, velocities = calculate_distance_velocity(galaxy_df)

# พล็อตกราฟความสัมพันธ์ระหว่างระยะทางและความเร็ว
plt.figure(figsize=(8, 6))
plt.scatter(distances, velocities, color='blue', label='Galaxy Data')
plt.title('Relationship between Distance (d) and Velocity (v)')
plt.xlabel('Distance (parsecs)')
plt.ylabel('Velocity (km/s)')
plt.legend()
plt.grid(True)
plt.savefig("distance_velocity_scatter.png")  # บันทึกกราฟเป็นไฟล์

# ฟิตเส้นตรงตามกฎของฮับเบิลและพล็อตเส้น predicted line
model = LinearRegression()
distances_reshape = distances.reshape(-1, 1)
model.fit(distances_reshape, velocities)
predicted_velocities = model.predict(distances_reshape)

plt.figure(figsize=(8, 6))
plt.scatter(distances, velocities, color='blue', label='Galaxy Data')
plt.plot(distances, predicted_velocities, color='red', label='Predicted Line (Hubble\'s Law)')
plt.title('Hubble\'s Law: Distance vs Velocity')
plt.xlabel('Distance (parsecs)')
plt.ylabel('Velocity (km/s)')
plt.legend()
plt.grid(True)
plt.savefig("hubble_law_fit.png")  # บันทึกกราฟเป็นไฟล์

# คำนวณค่า Hubble Constant และค่าความคลาดเคลื่อน
hubble_constant = model.coef_[0]
hubble_error = np.std(velocities - predicted_velocities)

print(f"Hubble Constant (H): {hubble_constant:.10f} km/s per parsec")
print(f"Error in Hubble Constant: {hubble_error:.2f} km/s per parsec")

# คำนวณอายุของเอกภพโดยประมาณ
H_in_s = hubble_constant * (1 / 3.086e19)  # แปลงจาก km/s/parsec เป็น 1/s
age_of_universe = 1 / H_in_s  # หน่วยเป็นวินาที
age_of_universe_years = age_of_universe / (60 * 60 * 24 * 365.25)

print(f"Estimated Age of Universe: {age_of_universe_years:.2e} years")