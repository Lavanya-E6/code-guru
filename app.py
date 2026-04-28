import streamlit as st
import torch
import sys
import io
import traceback
from transformers import AutoTokenizer, AutoModelForCausalLM

st.set_page_config(
    page_title="High-Perf Code Mentor AI",
    page_icon="⚡",
    layout="wide",
)

st.markdown("""
<style>
    .stTextArea textarea {
        font-family: 'Courier New', Courier, monospace;
        font-size: 14px;
        background-color: #1e1e1e;
        color: #d4d4d4;
        border: 1px solid #333;
    }
    .stButton>button {
        width: 100%; 
        border-radius: 6px;
        font-weight: 600;
        height: 50px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        border-color: #4CAF50;
        color: #4CAF50;
    }
    .output-box { background-color: #f0f2f6; padding: 15px; border-radius: 8px; border-left: 5px solid #4CAF50; font-family: monospace; white-space: pre-wrap; color: #333;}
    .error-box { background-color: #ffeaea; padding: 15px; border-radius: 8px; border-left: 5px solid #f44336; color: #d32f2f; font-family: monospace; white-space: pre-wrap;}
    .fix-box { background-color: #e8f5e9; padding: 15px; border-radius: 8px; border-left: 5px solid #4CAF50; color: #2e7d32; white-space: pre-wrap;}
</style>
""", unsafe_allow_html=True)

st.title("⚡ High-Performance Hybrid Code Mentor")
st.markdown("Instantly run code, detect errors locally, and use optimized LLM purely for explanations.")

# Ultra-lightweight coder model for instantaneous response
MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B"

@st.cache_resource(show_spinner=False)
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Use bfloat16 if supported for better speed/precision balance
    if device == "cuda":
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        torch_dtype = torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True
    ).to(device)
    
    return tokenizer, model, device

with st.spinner(f"Loading lightweight model ({MODEL_NAME})..."):
    try:
        tokenizer, model, device = load_model()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

# --- Sidebar / Settings ---
st.sidebar.header("⚙️ Settings")
mode_toggle = st.sidebar.radio("Optimization Mode", ["Fast Mode (Default)", "Detailed Mode"])
is_fast_mode = "Fast" in mode_toggle

# Adjust parameters based on mode
MAX_TOKENS = 250 if is_fast_mode else 750
TEMPERATURE = 0.1 if is_fast_mode else 0.4

# --- Helper logic for local execution ---
def execute_python_code(code: str, user_inputs: str = ""):
    """Executes Python code safely, returns (output, error_message, error_line)"""
    import time
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    redirected_output = sys.stdout = io.StringIO()
    redirected_error = sys.stderr = io.StringIO()
    
    error_msg = None
    error_line = None
    
    input_iterator = iter(user_inputs.splitlines())
    def custom_input(prompt=""):
        if prompt:
            redirected_output.write(str(prompt))
        try:
            return next(input_iterator)
        except StopIteration:
            raise EOFError("EOF when reading a line (no more input provided)")
            
    # Include default builtins so standard functions work, overriding input
    exec_globals = {"__builtins__": __builtins__, "input": custom_input}
    
    start_time = time.time()
    timeout_seconds = 5.0
    
    def trace_calls(frame, event, arg):
        if time.time() - start_time > timeout_seconds:
            raise TimeoutError(f"Execution exceeded the {timeout_seconds} seconds limit.")
        return trace_calls
        
    sys.settrace(trace_calls)
    try:
        exec(code, exec_globals)
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb = traceback.extract_tb(exc_traceback)
        
        for frame in reversed(tb):
            if frame.filename == "<string>":
                error_line = frame.lineno
                break
        
        error_msg = f"{exc_type.__name__}: {str(e)}"
        
    finally:
        sys.settrace(None)
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
    output = redirected_output.getvalue() + redirected_error.getvalue()
    return output.strip(), error_msg, error_line

def get_instruction_for_mode(action: str, code_length: int, error_context: dict = None) -> str:
    """Return an optimized, short prompt based on user's selected mode and text."""
    if action == "Explain Code":
        if code_length > 500:
            return "Summarize the following Python code in exactly 5-7 short bullet points. Explain its core purpose."
        else:
            if is_fast_mode:
                return "Briefly explain the following Python code in 3-5 bullet points. Keep it short."
            return "Explain the following Python code line by line clearly."
            
    elif action == "Fix Errors":
        err_msg = error_context.get("message", "Unknown error")
        err_line = error_context.get("line", "Unknown")
        return f"Python Error at line {err_line}: {err_msg}. Explain why this happened and provide the corrected code snippet briefly."

    elif action == "Analyze Complexity":
        if is_fast_mode:
            return "State only the Time Complexity (Big-O) and Space Complexity of the following Python code. Do not explain."
        return "Analyze the Time (Big-O) and Space Complexity of the following Python code. Keep explanation concise."
        
    elif action == "Optimize Code":
        return "Provide a short, optimized version of the following Python code. List improvements in max 3 bullet points."
        
    return "Analyze the code."

@st.cache_data(show_spinner=False, max_entries=200)
def generate_response(prompt_text: str, _tokenizer, _model, _device, max_tokens: int, temperature: float):
    """Generate response from the loaded LLM with aggressive limits. Cached for instant responses (anti-gravity feel)."""
    # Simple prompt formatting
    prompt = f"Instruction: {prompt_text}\nOutput:"
    
    inputs = _tokenizer(prompt, return_tensors="pt").to(_device)
    
    # torch.inference_mode() is faster than no_grad()
    with torch.inference_mode():
        outputs = _model.generate(
            **inputs, 
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=(temperature > 0),
            pad_token_id=_tokenizer.eos_token_id
        )
        
    input_length = inputs["input_ids"].shape[1]
    response = _tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    
    return response.strip()

# --- UI Setup ---
code_col, input_col = st.columns([3, 1])

with code_col:
    code_input = st.text_area(
        "Code Input",
        height=300, 
        placeholder="# Paste your Python code here...",
        label_visibility="collapsed"
    )

with input_col:
    program_input = st.text_area(
        "Program Input",
        height=300,
        placeholder="Enter program input here...\nEach value on a new line.\n\nExample:\n10\n20",
    )

col1, col2, col3, col4, col5 = st.columns(5)

action = None

if col1.button("▶️ Run Code", type="primary", use_container_width=True):
    action = "Run Code"
if col2.button("🧠 Explain", use_container_width=True):
    action = "Explain Code"
if col3.button("🛠️ Fix Errors", use_container_width=True):
    action = "Fix Errors"
if col4.button("⏱️ Complexity", use_container_width=True):
    action = "Analyze Complexity"
if col5.button("⚡ Optimize", use_container_width=True):
    action = "Optimize Code"

# --- Handle Execution / Generation ---
if action:
    if not code_input.strip():
        st.warning("⚠️ Please paste some code first.")
    else:
        st.markdown("---")
        
        if action == "Run Code":
            st.subheader("Console Output")
            with st.spinner("Executing instantly..."):
                output, error_msg, error_line = execute_python_code(code_input, program_input)
            
            if error_msg:
                st.markdown(f"**Error detected at line {error_line}:**")
                lines = code_input.split('\\n')
                if error_line and 1 <= error_line <= len(lines):
                    err_snippet = lines[error_line - 1]
                    st.markdown(f"<div class='error-box'>Line {error_line}: {err_snippet}<br><br>{error_msg}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='error-box'>{error_msg}</div>", unsafe_allow_html=True)
                if output:
                    st.markdown(f"**Standard Output:**<br><div class='output-box'>{output}</div>", unsafe_allow_html=True)
            else:
                if output:
                    st.markdown(f"<div class='output-box'>{output}</div>", unsafe_allow_html=True)
                else:
                    st.success("Code executed successfully (No output).")
                    
        else:
            st.subheader(f"Result: {action}")
            if action == "Fix Errors":
                output, error_msg, error_line = execute_python_code(code_input, program_input)
                if not error_msg:
                    st.success("No errors detected by the local Python runtime! 🎉")
                else:
                    lines = code_input.split('\\n')
                    err_snippet = lines[error_line - 1] if error_line and 1 <= error_line <= len(lines) else code_input[:200]
                    
                    st.markdown(f"<div class='error-box'><b>Detected:</b> {error_msg} at line {error_line}</div><br>", unsafe_allow_html=True)
                    
                    prompt_text = get_instruction_for_mode(action, len(code_input), {"message": error_msg, "line": error_line})
                    prompt_text += f"\n\nCode snippet causing error:\n```python\n{err_snippet}\n```"
                    
                    with st.spinner("Asking LLM for a fast fix..."):
                        try:
                            result = generate_response(prompt_text, tokenizer, model, device, MAX_TOKENS, TEMPERATURE)
                            st.markdown(f"<div class='fix-box'><b>Suggested Fix:</b><br><br>{result}</div>", unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"LLM Error: {str(e)}")
                        
            else:
                prompt_text = get_instruction_for_mode(action, len(code_input))
                prompt_text += f"\n\nCode to analyze:\n```python\n{code_input[:3000]}\n```" 
                with st.spinner(f"Analyzing at maximum speed... ({'Fast' if is_fast_mode else 'Detailed'} Mode)"):
                    try:
                        result = generate_response(prompt_text, tokenizer, model, device, MAX_TOKENS, TEMPERATURE)
                        st.markdown(result)
                    except Exception as e:
                        st.error(f"LLM Error: {str(e)}")
