# การวิเคราะห์ปัจจัยที่มีผลต่อราคาทองคำ (Gold Price Analysis)

เอกสารนี้อธิบายกระบวนการทำงานของโค้ด Python ในไฟล์ `goldprice_eda_06.ipynb` โดยเน้นเหตุผลของการกระทำ (Why), การเลือกเครื่องมือ (Tools Selection), และการตีความผลลัพธ์ (Interpretation)

## 1\. การเตรียมสภาพแวดล้อม (Environment Setup)

### สิ่งที่โค้ดทำ

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
...
sns.set_theme(style="whitegrid")
```

### เหตุผลและการเลือกเครื่องมือ

  * **Pandas:** เป็น Library มาตรฐานสำหรับ **Data Manipulation** ใช้จัดการข้อมูลในรูปแบบตาราง (DataFrame) ซึ่งจำเป็นมากสำหรับการจัดการ Time-series financial data
  * **Seaborn / Matplotlib:** ใช้สำหรับ **Data Visualization** โดยเลือก `seaborn` เพื่อความสวยงามและ `matplotlib` เพื่อการปรับแต่งรายละเอียดกราฟ (เช่น การทำ 2 แกน Y)
  * **`functools.reduce`:** ใช้เพื่อรวม DataFrames หลายตารางเข้าด้วยกันอย่างมีประสิทธิภาพในคำสั่งเดียว

-----

## 2\. การรวบรวมและทำความสะอาดข้อมูล (Data Preprocessing & Cleaning)

### สิ่งที่โค้ดทำ (`load_and_fix_data`)

1.  โหลดข้อมูลดิบ (Gold, Bitcoin, FX, Oil)
2.  จัดการรูปแบบวันที่ (Date parsing) ให้เป็น format เดียวกัน
3.  **Feature Logic:** แปลงราคา Bitcoin ที่เป็น USD ให้เป็น THB โดยเช็คเงื่อนไข `btc_price < 500,000`
4.  **Merge:** รวมตารางทั้งหมดด้วยวันที่ (`outer join`)
5.  **Imputation:** เรียงวันที่และเติมค่าที่หายไปแบบ `ffill` (Forward Fill)

### การวิเคราะห์เชิง Data Science & Economics

  * **Consistency (ความสม่ำเสมอ):** สาเหตุที่ต้องแปลงราคา Bitcoin เป็น THB เนื่องจากข้อมูลดิบมีหน่วยผสมกัน (USD และ THB) ตามหลักเศรษฐศาสตร์ การเปรียบเทียบมูลค่าสินทรัพย์ต้องอยู่บน **Unit of Account** (หน่วยวัดมูลค่า) เดียวกันเพื่อให้เห็นความสัมพันธ์ที่แท้จริง
  * **Missing Data Handling (ffill):** ในข้อมูลทางการเงิน (Financial Time Series) ตลาดแต่ละแห่งหยุดไม่พร้อมกัน (เช่น ทองหยุดเสาร์-อาทิตย์ แต่ Bitcoin เทรด 24/7)
      * *ทำไมต้อง Forward Fill:* เราสมมติว่า **"ราคาล่าสุดยังคงมีผลอยู่จนกว่าจะมีการซื้อขายใหม่"** วิธีนี้ดีกว่าการเติม 0 หรือค่าเฉลี่ย ซึ่งจะทำให้กราฟราคาผิดเพี้ยนไปจากความจริง
  * **Data Filtering:** การตัดข้อมูลเหลือแค่ปี 2023 - ปัจจุบัน (Recent Trend) เพื่อลดผลกระทบจากเหตุการณ์ในอดีตที่โครงสร้างตลาดอาจต่างจากปัจจุบัน (Structural Break)

-----

## 3\. การวิเคราะห์แนวโน้มด้วยกราฟสองแกน (Dual-Axis Time Series Plot)

### สิ่งที่โค้ดทำ (`plot_trend_comparison`)

สร้างกราฟเส้นที่มีแกน Y สองฝั่ง ฝั่งซ้ายเป็นราคาทองคำ (Gold) และฝั่งขวาเป็นตัวแปรคู่เทียบ (Bitcoin หรือ FX)

### เหตุผลและการเลือกเครื่องมือ

  * **Matplotlib (`twinx`):** ใช้เทคนิค **Dual Axis** เพราะข้อมูลสองชุดมี **Scale ต่างกันมหาศาล** (Gold หลักหมื่นบาท vs Bitcoin หลักล้านบาท หรือ FX หลักสิบบาท) หากพล็อตกราฟแกนเดียวกัน เส้นใดเส้นหนึ่งจะแบนติดพื้นจนมองไม่เห็นแนวโน้ม

### การวิเคราะห์เชิง Statistics & Economics

  * **Trend Analysis:** เราต้องการดู **Co-movement** (การเคลื่อนที่ไปด้วยกัน) ไม่ใช่มูลค่าสัมบูรณ์
  * **Gold vs Bitcoin:**
      * *สิ่งที่เห็น:* ทั้งคู่มีแนวโน้มเป็นขาขึ้น (Upward Trend) ในช่วงเวลาเดียวกัน
      * *Economic Interpretation:* ทั้งทองคำและ Bitcoin มักถูกมองว่าเป็น **Store of Value** หรือสินทรัพย์ป้องกันความเสี่ยง (Hedge) ในสภาวะที่เงินเฟ้อสูงหรือค่าเงินอ่อนค่า นักลงทุนอาจกระจายเงินเข้าทั้งสองสินทรัพย์นี้พร้อมๆ กัน
  * **Gold vs FX (USD/THB):**
      * *สิ่งที่เห็น:* ราคาทองคำในไทยเคลื่อนไหวไปในทิศทางเดียวกับค่าเงินบาท/ดอลลาร์
      * *Economic Interpretation:* ราคาทองคำโลก (Gold Spot) ซื้อขายเป็น USD เมื่อแปลงเป็นราคาทองไทย หากเงินบาทอ่อนค่า (USD/THB สูงขึ้น) ราคาทองไทยจะสูงขึ้นทันที แม้ราคาทองโลกจะเท่าเดิม นี่คือ **Exchange Rate Pass-through effect**

-----

## 4\. การวิเคราะห์ความสัมพันธ์ (Correlation Analysis via Regression Plot)

### สิ่งที่โค้ดทำ (`sns.regplot`)

สร้างกราฟ Scatter Plot พร้อมเส้น Regression Line เพื่อดูความสัมพันธ์ระหว่างตัวแปร

### เหตุผลและการเลือกเครื่องมือ

  * **Seaborn (`regplot`):** เป็นเครื่องมือที่ดีที่สุดในการทำ **Bivariate Analysis** (วิเคราะห์สองตัวแปร) เพื่อดูว่าเมื่อตัวแปร X เปลี่ยน ตัวแปร Y เปลี่ยนไปอย่างไร
  * **Scatter + Line:** จุดแสดงการกระจายตัว (Variance) ส่วนเส้นแสดงแนวโน้มความสัมพันธ์ (Linear Relationship)

### การวิเคราะห์เชิง Statistics & Data Science

  * **Gold vs Bitcoin Correlation (รูปซ้าย):**
      * *Stat:* จุดเกาะกลุ่มกันเป็นแนวเฉียงขึ้น (Positive Correlation) แต่มีการกระจายตัว (Variance) สูงในบางช่วง
      * *Insight:* มีความสัมพันธ์เชิงบวก แต่อาจไม่แข็งแรงเท่ากับปัจจัยอื่น Bitcoin มีความผันผวน (Volatility) สูงกว่าทองคำมาก
  * **Gold vs FX Correlation (รูปขวา):**
      * *Stat:* จุดเรียงตัวชิดเส้น Regression มากกว่า (Tight fit) แสดงถึง **Strong Positive Correlation**
      * *Insight:* ค่าเงินบาท (USD/THB) เป็น **Predictor** (ตัวทำนาย) ที่สำคัญมากต่อราคาทองคำไทย ความสัมพันธ์นี้เป็นเชิงเส้น (Linear) ที่ชัดเจน
      * *Data Sci Action:* ในการสร้างโมเดลทำนายราคาทอง (Machine Learning Model) ตัวแปร `fx` (อัตราแลกเปลี่ยน) จะเป็น Feature ที่มีน้ำหนักความสำคัญ (Feature Importance) สูงมาก

-----

### สรุปภาพรวม (Conclusion)

จากการวิเคราะห์ด้วย EDA (Exploratory Data Analysis) เบื้องต้น:

1.  **ราคาทองคำไทยผูกพันกับค่าเงินบาทสูงมาก:** หากจะทำนายราคาทอง ต้องแม่นยำเรื่องทิศทางค่าเงิน
2.  **Bitcoin และทองคำมีทิศทางคล้ายกัน:** อาจใช้เป็น Indicator เสริมในแง่ของ Sentiment ตลาดสินทรัพย์ทางเลือก (Alternative Assets) แต่ความผันผวนของ Bitcoin ทำให้เป็นสัญญาณที่มี Noise เยอะกว่าค่าเงิน