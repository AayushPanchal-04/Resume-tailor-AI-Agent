import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Resume Tailor Agent", page_icon="📄", layout="wide")

st.title("📄 Resume Tailor Agent")
st.markdown("### Automatically optimize your resume for any job description using AI")

with st.sidebar:
    st.header("Configuration")
    
  
    default_api_key = os.getenv("OPENAI_API_KEY", "")
    
    
    api_key = st.text_input(
        "OpenAI API Key", 
        value=default_api_key if default_api_key else "",
        type="password", 
        help="Get your API key from platform.openai.com or set OPENAI_API_KEY in .env file"
    )
    
    if default_api_key:
        st.success("API key loaded from environment")
    
    st.markdown("---")
    st.markdown("### How it works:")
    st.markdown("""
    1. Enter your OpenAI API key
    2. Paste your current resume
    3. Paste the job description
    4. Click 'Tailor Resume'
    5. Get an optimized version!
    """)
    st.markdown("---")
    st.markdown("### Tips:")
    st.markdown("""
    - Use complete resume text
    - Include full job description
    - Review and edit the output
    - Keep your authentic experience
    """)


col1, col2 = st.columns(2)

with col1:
    st.subheader("Your Current Resume")
    resume_text = st.text_area(
        "Paste your resume here",
        height=400,
        placeholder="Paste your complete resume text here..."
    )

with col2:
    st.subheader("Target Job Description")
    job_description = st.text_area(
        "Paste the job description",
        height=400,
        placeholder="Paste the job description you're applying for..."
    )


if st.button("Tailor Resume", type="primary", use_container_width=True):
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar")
    elif not resume_text or not job_description:
        st.error("Please provide both your resume and the job description")
    else:
        with st.spinner("AI is tailoring your resume..."):
            try:
                
                llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0.7,
                    api_key=api_key
                )
                
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are an expert resume writer and career consultant. Your task is to tailor resumes to match job descriptions while maintaining authenticity.

Guidelines:
1. Analyze the job description to identify key skills, requirements, and keywords
2. Reorganize and rewrite the resume to highlight relevant experience
3. Use action verbs and quantifiable achievements
4. Match the tone and language of the job description
5. Ensure all information remains truthful - only reframe existing experience
6. Optimize for ATS (Applicant Tracking Systems) by including relevant keywords
7. Keep the same format structure but improve content clarity
8. Focus on impact and results

Important: Only use information from the original resume. Do not fabricate experience."""),
                    ("user", """Job Description:
{job_description}

Current Resume:
{resume}

Please tailor this resume to match the job description. Provide a complete, polished resume that emphasizes relevant experience and skills.""")
                ])
                
                
                chain = prompt | llm | StrOutputParser()
                
               
                tailored_resume = chain.invoke({
                    "job_description": job_description,
                    "resume": resume_text
                })
                
                
                st.success("Resume tailored successfully!")
                st.markdown("---")
                st.subheader("Your Tailored Resume")
                st.markdown(tailored_resume)
                
              
                st.download_button(
                    label="Download Tailored Resume",
                    data=tailored_resume,
                    file_name="tailored_resume.txt",
                    mime="text/plain"
                )
                
                
                with st.expander("View Optimization Analysis"):
                    analysis_prompt = ChatPromptTemplate.from_messages([
                        ("system", "You are a resume analyst. Compare the original and tailored resumes and explain the key improvements."),
                        ("user", """Original Resume:
{original}

Tailored Resume:
{tailored}

Job Description:
{job_desc}

Provide a brief analysis of:
1. Key changes made
2. Skills and keywords emphasized
3. How it better matches the job description""")
                    ])
                    
                    analysis_chain = analysis_prompt | llm | StrOutputParser()
                    analysis = analysis_chain.invoke({
                        "original": resume_text,
                        "tailored": tailored_resume,
                        "job_desc": job_description
                    })
                    st.markdown(analysis)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Make sure your API key is valid and you have credits available")


st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with LangChain 🦜🔗 and Streamlit</p>
    <p style='font-size: 0.8em; color: gray;'>Remember: Always review and verify the tailored resume before submitting!</p>
</div>
""", unsafe_allow_html=True)