import xgboost as xgb
import streamlit as st
import pandas as pd
import time
import google.generativeai as genai
import os

# Page configuration
st.set_page_config(
    page_title="DiamondGenius",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Main color scheme */
    :root {
        --primary-color: #1e3a8a;
        --secondary-color: #3b82f6;
        --accent-color: #f59e0b;
        --dark-bg: #0f172a;
        --light-bg: #1e293b;
        --text-color: #e2e8f0;
    }
    
    /* Base styles */
    .stApp {
        background-color: var(--dark-bg);
        color: var(--text-color);
    }
    
    h1, h2, h3 {
        color: var(--text-color);
        font-family: 'Poppins', sans-serif;
    }
    
    /* Card style */
    .card {
        background-color: var(--light-bg);
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Button styling */
    .stButton > button {
        background-color: var(--accent-color);
        color: #000000;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 10px 25px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #d97706;
        transform: scale(1.05);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: var(--light-bg);
        border-radius: 5px 5px 0 0;
        padding: 10px 20px;
        color: var(--text-color);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--secondary-color);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: var(--light-bg);
        padding: 20px;
    }
    
    /* Success message styling */
    .stSuccess {
        background-color: #065f46;
        color: white;
        padding: 20px;
        border-radius: 10px;
        font-weight: bold;
        animation: fadeIn 0.5s ease-in;
    }
    
    /* Slider styling */
    [data-testid="stSlider"] > div {
        color: var(--text-color);
    }
    
    /* Animation keyframes */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes slideIn {
        from { transform: translateX(-20px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    /* Apply animations */
    .animate-fade {
        animation: fadeIn 0.6s ease-in;
    }
    
    .animate-slide {
        animation: slideIn 0.6s ease-out;
    }
    
    /* Select box styling */
    .stSelectbox {
        color: var(--text-color);
    }
    
    /* Diamond icon animation */
    .diamond-icon {
        display: inline-block;
        font-size: 3rem;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); opacity: 0.8; }
        50% { transform: scale(1.1); opacity: 1; }
        100% { transform: scale(1); opacity: 0.8; }
    }
    
    /* Chat message styling */
    .chat-message {
        margin-bottom: 10px;
        padding: 10px 15px;
        border-radius: 10px;
        max-width: 80%;
    }
    
    .user-message {
        background-color: var(--primary-color);
        margin-left: auto;
        border-bottom-right-radius: 0;
    }
    
    .assistant-message {
        background-color: var(--secondary-color);
        margin-right: auto;
        border-bottom-left-radius: 0;
    }
    
    /* Add responsiveness */
    @media (max-width: 768px) {
        .card {
            padding: 15px;
        }
    }
    
    /* Custom image container */
    .img-container {
        display: flex;
        justify-content: center;
        margin: 10px 0;
    }
    
    .img-container img {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        max-width: 100%;
        height: auto;
    }
    
    /* Currency display styling */
    .currency-box {
        background-color: var(--primary-color);
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .currency-name {
        font-weight: bold;
        margin-bottom: 5px;
    }
    
    .currency-value {
        font-size: 1.2rem;
        font-weight: bold;
        color: var(--accent-color);
    }
</style>
""", unsafe_allow_html=True)

# Initialize Gemini API (hardcoded API key)
# In a production environment, this should be stored securely using environment variables
# or a secret management service
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]  # Replace with your actual API key
genai.configure(api_key=GEMINI_API_KEY)

# Function to initialize and configure Gemini model
@st.cache_resource
def init_gemini_model():
    # Configure the generative model with diamond-specific context
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    return model

# Loading up the Regression model
@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model('xgb_model.json')
    return model

model = load_model()
gemini_model = init_gemini_model()

# Define the prediction function
def predict(carat, cut, color, clarity, depth, table, x, y, z):
    # Mapping categorical variables
    cut_mapping = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
    color_mapping = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}
    clarity_mapping = {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7}
    
    # Use the mappings directly
    cut_encoded = cut_mapping[cut]
    color_encoded = color_mapping[color]
    clarity_encoded = clarity_mapping[clarity]
    
    # Create prediction dataframe
    prediction_df = pd.DataFrame([[carat, cut_encoded, color_encoded, clarity_encoded, depth, table, x, y, z]], 
                                columns=['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z'])
    
    # Make prediction
    prediction = model.predict(prediction_df)
    return prediction

# Function to convert USD to other currencies
def convert_currencies(usd_price):
    # Exchange rates (as of March 2025 - for simulation purposes)
    exchange_rates = {
        'INR': 83.5,  # 1 USD = 83.5 INR
        'JPY': 149.8, # 1 USD = 149.8 JPY
        'AED': 3.67   # 1 USD = 3.67 AED
    }
    
    return {
        'USD': usd_price,
        'INR': usd_price * exchange_rates['INR'],
        'JPY': usd_price * exchange_rates['JPY'],
        'AED': usd_price * exchange_rates['AED']
    }

# Function to generate diamond insights based on characteristics
def generate_diamond_insights(carat, cut, color, clarity):
    insights = {
        "Fair": "A fair cut reflects less light, resulting in less brilliance. While economical, this cut doesn't showcase a diamond's potential sparkle.",
        "Good": "Good cuts offer decent brilliance at a reasonable price point, making them suitable for budget-conscious buyers who still want quality.",
        "Very Good": "Very Good cuts provide excellent brilliance and fire. They're an excellent value, offering nearly the same visual appeal as Ideal cuts at a lower price.",
        "Premium": "Premium cuts display exceptional brilliance and fire. They're precision-cut to maximize light reflection, though sometimes with slightly deeper proportions.",
        "Ideal": "Ideal cuts represent the pinnacle of diamond cutting, with perfect proportions to maximize brilliance and fire. They reflect nearly all light that enters the diamond."
    }
    
    color_insights = {
        "D": "Completely colorless and extremely rare. The highest color grade available.",
        "E": "Colorless, but slightly less rare than D. Differences are not visible to the untrained eye.",
        "F": "Colorless, but detectable by gemologists. Still considered colorless to the naked eye.",
        "G": "Near-colorless with slight traces of color, visible only to expert gemologists.",
        "H": "Near-colorless with minimal color visible under magnification.",
        "I": "Near-colorless with slight warmth that may be visible in larger diamonds.",
        "J": "Near-colorless with noticeable warmth that provides good value."
    }
    
    clarity_insights = {
        "IF": "Internally Flawless: No internal inclusions visible under 10x magnification.",
        "VVS1": "Very, Very Slightly Included 1: Contains minute inclusions difficult for expert gemologists to see.",
        "VVS2": "Very, Very Slightly Included 2: Contains minute inclusions slightly easier to see than VVS1.",
        "VS1": "Very Slightly Included 1: Contains minor inclusions difficult to see under 10x magnification.",
        "VS2": "Very Slightly Included 2: Contains minor inclusions visible under 10x magnification.",
        "SI1": "Slightly Included 1: Contains noticeable inclusions under 10x magnification but often not visible to naked eye.",
        "SI2": "Slightly Included 2: Contains noticeable inclusions under 10x magnification, sometimes visible to the naked eye.",
        "I1": "Included 1: Contains inclusions visible to the naked eye that may affect brilliance."
    }
    
    carat_insight = f"At {carat} carats, this diamond has significant presence. "
    if carat < 0.5:
        carat_insight = f"At {carat} carats, this diamond is delicate and subtle. "
    elif carat < 1.0:
        carat_insight = f"At {carat} carats, this diamond has a good balance of presence and value. "
    elif carat < 2.0:
        carat_insight = f"At {carat} carats, this diamond makes a substantial statement. "
    else:
        carat_insight = f"At {carat} carats, this diamond has exceptional presence and rarity. "
    
    return {
        "cut": insights.get(cut, ""),
        "color": color_insights.get(color, ""),
        "clarity": clarity_insights.get(clarity, ""),
        "carat": carat_insight
    }

# Function to generate expert response using Gemini API
def generate_expert_response(prompt):
    try:
        # Create a system prompt that guides Gemini to act as a diamond expert
        system_prompt = """
        You are DiamondGenius, an expert AI advisor specializing in diamonds. Provide accurate, helpful information about:
        - Diamond quality factors (4Cs: Cut, Color, Clarity, Carat)
        - Pricing considerations and market trends
        - Diamond investment advice
        - Ethical considerations and lab-grown diamonds
        - Diamond maintenance and care
        - Diamond price estimation
        - Diamond shopping tips
        
        Keep responses concise (under 250 words) yet informative. Use formal but accessible language.
        Always provide balanced information, considering both traditional and modern perspectives on diamonds.
        """
        
        # Create a safety context for the model
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
        
        # Combine the system prompt with the user's query
        full_prompt = f"{system_prompt}\n\nUser question: {prompt}"
        
        # Generate response from Gemini
        response = gemini_model.generate_content(
            full_prompt,
            generation_config={"temperature": 0.2, "max_output_tokens": 500}
        )
        
        return response.text
    except Exception as e:
        # Fallback response in case of API errors
        return f"I apologize, but I'm having trouble connecting to my knowledge base at the moment. Please try again in a few moments. (Error: {str(e)})"

# Sidebar content
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: #3b82f6;'>💎 Adamas AI</h1>", unsafe_allow_html=True)
    
    # Diamond logo/animation (CSS-based instead of Lottie)
    st.markdown("<div style='text-align: center;'><span class='diamond-icon'>💎</span></div>", unsafe_allow_html=True)
    
    
    st.markdown("<div class='card animate-fade'>", unsafe_allow_html=True)
    st.markdown("### About Adamas AI")
    st.write("Adamas AI is your smart companion for diamond valuation and knowledge. Using advanced machine learning, we provide accurate price predictions and expert advice.")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='card animate-fade'>", unsafe_allow_html=True)
    st.markdown("### Developer Info")
    st.write("Developed by Diamond Analytics Inc.")
    st.write("© 2025 All Rights Reserved")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Simulate loading
    if st.sidebar.button("Refresh Data"):
        with st.spinner("Refreshing data..."):
            time.sleep(1.5)
        st.success("Data refreshed successfully!")

# Main content with tabs
tab1, tab2, tab3 = st.tabs(["💼 Quality Analysis", "📚 About Diamonds", "🤖 Expert Advice"])

# Tab 1: Quality Analysis
with tab1:
    st.markdown("<h1 class='animate-slide'>Diamond Price Predictor</h1>", unsafe_allow_html=True)
    
    # Diamond analysis icon
    st.markdown("""
    <div style="text-align: right;">
        <span style="font-size: 3rem; color: #3b82f6;">💎📊</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='animate-fade'>", unsafe_allow_html=True)
    st.markdown("### Enter the characteristics of your diamond to get an accurate price prediction.")
    st.markdown("Our advanced AI model analyzes multiple factors to provide market-accurate valuations.")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='card animate-fade'>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Diamond Measurements")
        carat = st.slider("Carat Weight", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        depth = st.slider("Diamond Depth Percentage", min_value=45.0, max_value=75.0, value=61.5, step=0.1)
        table = st.slider("Diamond Table Percentage", min_value=45.0, max_value=75.0, value=57.0, step=0.1)
        
        x = st.slider("Diamond Length (X) in mm", min_value=0.1, max_value=30.0, value=6.0, step=0.1)
        y = st.slider("Diamond Width (Y) in mm", min_value=0.1, max_value=30.0, value=6.0, step=0.1)
        z = st.slider("Diamond Height (Z) in mm", min_value=0.1, max_value=30.0, value=4.0, step=0.1)
    
    with col2:
        st.markdown("### Diamond Quality")
        cut = st.selectbox('Cut Rating', ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
        color = st.selectbox('Color Rating', ['J', 'I', 'H', 'G', 'F', 'E', 'D'])
        clarity = st.selectbox('Clarity Rating', ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])
    
    if st.button('Predict Diamond Price'):
        with st.spinner("Analyzing diamond characteristics..."):
            time.sleep(1)  # For dramatic effect
            price = predict(carat, cut, color, clarity, depth, table, x, y, z)
            
            # Convert price to multiple currencies
            price_value = price[0]
            currencies = convert_currencies(price_value)
        
        # Display price in multiple currencies
        st.markdown("### Diamond Valuation")
        
        currency_cols = st.columns(4)
        
        with currency_cols[0]:
            st.markdown("<div class='currency-box'>", unsafe_allow_html=True)
            st.markdown("<div class='currency-name'>USD (US Dollar)</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='currency-value'>${currencies['USD']:,.2f}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with currency_cols[1]:
            st.markdown("<div class='currency-box'>", unsafe_allow_html=True)
            st.markdown("<div class='currency-name'>INR (Indian Rupee)</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='currency-value'>₹{currencies['INR']:,.2f}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with currency_cols[2]:
            st.markdown("<div class='currency-box'>", unsafe_allow_html=True)
            st.markdown("<div class='currency-name'>JPY (Japanese Yen)</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='currency-value'>¥{currencies['JPY']:,.0f}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with currency_cols[3]:
            st.markdown("<div class='currency-box'>", unsafe_allow_html=True)
            st.markdown("<div class='currency-name'>AED (UAE Dirham)</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='currency-value'>د.إ{currencies['AED']:,.2f}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Display insights
        insights = generate_diamond_insights(carat, cut, color, clarity)
        
        st.markdown("### Diamond Insights")
        insights_cols = st.columns(2)
        
        with insights_cols[0]:
            st.markdown(f"**Carat Assessment**: {insights['carat']}")
            st.markdown(f"**Cut Quality**: {insights['cut']}")
        
        with insights_cols[1]:
            st.markdown(f"**Color Grade**: {insights['color']}")
            st.markdown(f"**Clarity Assessment**: {insights['clarity']}")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Tab 2: About Diamonds
with tab2:
    st.markdown("<h1 class='animate-slide'>Diamond Knowledge Center</h1>", unsafe_allow_html=True)
    
    # Diamond info icon
    st.markdown("""
    <div style="text-align: right;">
        <span style="font-size: 3rem; color: #3b82f6;">💎📚</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='card animate-fade'>", unsafe_allow_html=True)
    st.markdown("## The 4 Cs of Diamonds")
    
    # Custom image for the 4Cs section
    st.markdown("""
    <div class="img-container">
        <img src="https://via.placeholder.com/800x300/1e293b/ffffff?text=Diamond+4Cs+Diagram" alt="Diamond 4Cs">
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    Diamonds are evaluated based on four main characteristics, known as the 4 Cs:
    
    ### 1. Carat
    Carat refers to the weight of a diamond, not its size. One carat equals 200 milligrams.
    - **Significance**: Larger diamonds are rarer and thus typically more valuable per carat.
    - **Common Sizes**: Engagement rings often feature diamonds between 0.5 and 2.0 carats.
    
    ### 2. Cut
    Cut refers to how well a diamond has been shaped and faceted, affecting its brilliance and sparkle.
    - **Fair**: Minimal light reflection, less sparkle
    - **Good**: Decent light reflection at a lower price point
    - **Very Good**: Excellent sparkle, nearly comparable to Ideal
    - **Premium**: Exceptional sparkle, sometimes with deeper proportions
    - **Ideal**: Maximum brilliance and fire, perfect proportions
    
    ### 3. Color
    Diamond color grading assesses the absence of color, with colorless diamonds being the most valuable.
    - **D-F**: Colorless (most valuable)
    - **G-J**: Near colorless
    - **K-M**: Faint yellow
    - **N-Z**: Very light to light yellow
    
    ### 4. Clarity
    Clarity measures the presence of inclusions and blemishes.
    - **FL/IF**: Flawless/Internally Flawless
    - **VVS1/VVS2**: Very, Very Slightly Included
    - **VS1/VS2**: Very Slightly Included
    - **SI1/SI2**: Slightly Included
    - **I1/I2/I3**: Included
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='card animate-fade'>", unsafe_allow_html=True)
    st.markdown("## Types of Diamonds and Their Uses")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Natural Diamonds
        Formed over billions of years deep within the Earth under extreme pressure and heat.
        
        **Uses**:
        - Fine jewelry and engagement rings
        - Status symbols and investments
        - Industrial cutting and grinding
        
        ### Lab-Grown Diamonds
        Chemically identical to natural diamonds but created in controlled laboratory environments.
        
        **Uses**:
        - Affordable alternative for jewelry
        - Ethical and environmentally conscious choice
        - Scientific and technological applications
        """)
    
    with col2:
        st.markdown("""
        ### Fancy Color Diamonds
        Natural diamonds with distinct colors like blue, pink, or yellow.
        
        **Uses**:
        - Collector's items and luxury jewelry
        - Ultra-high-end investments
        - Museum pieces
        
        ### Industrial Diamonds
        Lower quality diamonds used for their physical properties rather than appearance.
        
        **Uses**:
        - Cutting, grinding, and polishing tools
        - Thermal conductors in electronics
        - Medical equipment and scientific instruments
        """)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='card animate-fade'>", unsafe_allow_html=True)
    st.markdown("## Diamond Care and Maintenance")
    st.markdown("""
    ### Proper Cleaning
    - Soak diamonds in a solution of mild dish soap and warm water
    - Gently scrub with a soft toothbrush
    - Rinse thoroughly and pat dry with a lint-free cloth
    
    ### Storage
    - Store separately from other jewelry to prevent scratches
    - Keep in a fabric-lined jewelry box or individual pouches
    - Avoid exposure to household chemicals that can damage settings
    
    ### Professional Maintenance
    - Have professional cleaning every six months
    - Check prongs and settings annually for security
    - Insure valuable diamond pieces
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# Tab 3: Expert Advice
with tab3:
    st.markdown("<h1 class='animate-slide'>Diamond Expert Advisor</h1>", unsafe_allow_html=True)
    
    # Chatbot icon
    st.markdown("""
    <div style="text-align: right;">
        <span style="font-size: 3rem; color: #3b82f6;">💎💬</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='animate-fade'>", unsafe_allow_html=True)
    st.markdown("### Ask our AI diamond expert any questions about diamonds")
    st.markdown("Get instant advice on diamond selection, pricing factors, investment potential, and more.")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm the DiamondGenius AI advisor. Ask me anything about diamonds, from selection tips to investment advice!"}
        ]
    
    # Display chat messages
    st.markdown("<div class='card animate-fade'>", unsafe_allow_html=True)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about diamonds..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Consulting diamond expertise..."):
                # Generate expert response using Gemini API
                response_text = generate_expert_response(prompt)
                
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
    
    st.markdown("</div>", unsafe_allow_html=True)
