import os
import uuid
import logging
from typing import Dict, Any
from langgraph.graph import StateGraph, END

from backend.app.agents.state import AgentState
from backend.app.agents.supervisor import supervisor_node, supervisor_router, make_log_entry
from backend.app.agents.intake import intake_node
from backend.app.agents.preprocess import preprocess_document
from backend.app.agents.layout import analyze_layout
from backend.app.agents.ocr import extract_ocr_fields
from backend.app.agents.semantic import semantic_node
from backend.app.agents.validation import validation_node
from backend.app.agents.confidence import confidence_node
from backend.app.agents.human_review import human_review_node
from backend.app.agents.memory import memory_node

logger = logging.getLogger("graph_orchestrator")

# ==========================================
# WRAPPER NODES FOR STATE CONVERSION
# ==========================================

def preprocess_graph_node(state: AgentState) -> Dict[str, Any]:
    """Node wrapping OpenCV Preprocessing Agent."""
    file_path = state.get("file_path")
    params = state.get("preprocessing_params", {})
    doc_id = state.get("document_id")
    
    output_filename = f"preprocessed_{doc_id}_{os.path.basename(file_path)}"
    
    logs = [make_log_entry("Preprocess", f"Applying filters: {params}")]
    
    try:
        result = preprocess_document(
            image_path=file_path,
            output_filename=output_filename,
            deskew=params.get("deskew", True),
            denoise=params.get("denoise", True),
            clahe_enhancement=params.get("clahe_enhancement", False),
            adaptive_thresh=params.get("adaptive_thresh", False),
            scale_factor=params.get("scale_factor", 1.0)
        )
        
        logs.append(make_log_entry(
            "Preprocess", 
            f"Preprocessing completed. Actions taken: {result['actions_taken']}. Saved to static storage."
        ))
        
        return {
            "preprocessed_path": result["preprocessed_path"],
            "reasoning_log": logs
        }
    except Exception as e:
        logger.error(f"Preprocessing agent error: {e}")
        logs.append(make_log_entry("Preprocess", f"Preprocessing failed: {e}", level="ERROR"))
        return {
            "preprocessed_path": file_path,  # Use raw as fallback
            "reasoning_log": logs
        }

def layout_graph_node(state: AgentState) -> Dict[str, Any]:
    """Node wrapping Layout Intelligence Agent."""
    img_path = state.get("preprocessed_path") or state.get("file_path")
    logs = [make_log_entry("Layout", "Starting visual segmentation and template classification.")]
    
    try:
        # We pass the filename to assist template classification
        result = analyze_layout(img_path, document_text_sample=state.get("filename", ""))
        
        n_lines = len(result.get("horizontal_lines", []))
        logs.append(make_log_entry(
            "Layout", 
            f"Mapped template '{result['template_key']}' with {len(result['fields'])} input regions. "
            f"Detected {n_lines} horizontal form lines for bbox snapping."
        ))
        
        return {
            "template_key": result["template_key"],
            "fields": {
                f["key"]: {
                    "value": "", "confidence": 0.0,
                    "bbox": f["bbox"], "type": f["type"],
                    "label": f.get("label", f["key"]),
                    "extraction_method": f.get("extraction_method", "template_normalized"),
                }
                for f in result["fields"]
            },
            "reasoning_log": logs
        }
    except Exception as e:
        logger.error(f"Layout agent error: {e}")
        logs.append(make_log_entry("Layout", f"Layout mapping failed: {e}", level="ERROR"))
        return {"reasoning_log": logs}

def ocr_graph_node(state: AgentState) -> Dict[str, Any]:
    """Node wrapping Hybrid OCR Agent."""
    img_path = state.get("preprocessed_path") or state.get("file_path")
    fields = state.get("fields", {})
    # Read the template key set by the layout agent (critical for correct mock data selection)
    template_key = state.get("template_key", "membership_application")
    
    logs = [make_log_entry("OCR", "Initiating printed/handwritten text extraction on mapped bounding boxes.")]
    
    # Format layout format for the OCR agent
    layout_data = {
        "template_key": template_key,
        "fields": [
            {
                "key": k,
                "type": v.get("type", "printed_handwritten"),
                "bbox": v.get("bbox"),
                "extraction_method": v.get("extraction_method", "template_normalized"),
            }
            for k, v in fields.items()
        ]
    }
    
    try:
        ocr_result = extract_ocr_fields(img_path, layout_data)
        
        logs.append(make_log_entry(
            "OCR", 
            f"Extraction completed. Average confidence: {ocr_result['overall_ocr_confidence']:.2f}."
        ))
        
        return {
            "fields": ocr_result["fields"],
            "ocr_confidence": ocr_result["overall_ocr_confidence"],
            "reasoning_log": logs
        }
    except Exception as e:
        logger.error(f"OCR agent error: {e}")
        logs.append(make_log_entry("OCR", f"OCR extraction failed: {e}", level="ERROR"))
        return {"reasoning_log": logs}

# ==========================================
# BUILD THE GRAPH
# ==========================================

def compile_agent_workflow():
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("intake", intake_node)
    workflow.add_node("preprocess", preprocess_graph_node)
    workflow.add_node("layout", layout_graph_node)
    workflow.add_node("ocr", ocr_graph_node)
    workflow.add_node("semantic", semantic_node)
    workflow.add_node("validation", validation_node)
    workflow.add_node("confidence", confidence_node)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("memory", memory_node)
    
    # Set Entry Point
    workflow.set_entry_point("supervisor")
    
    # Set Routing Edges from Supervisor
    workflow.add_conditional_edges(
        "supervisor",
        supervisor_router,
        {
            "intake": "intake",
            "preprocess": "preprocess",
            "layout": "layout",
            "ocr": "ocr",
            "semantic": "semantic",
            "validation": "validation",
            "confidence": "confidence",
            "human_review": "human_review",
            "memory": "memory",
            "end": END
        }
    )
    
    # Connect Sub-agents back to Supervisor
    workflow.add_edge("intake", "supervisor")
    workflow.add_edge("preprocess", "supervisor")
    workflow.add_edge("layout", "supervisor")
    workflow.add_edge("ocr", "supervisor")
    workflow.add_edge("semantic", "supervisor")
    workflow.add_edge("validation", "supervisor")
    workflow.add_edge("confidence", "supervisor")
    workflow.add_edge("human_review", "supervisor")
    workflow.add_edge("memory", "supervisor")
    
    # Compile
    app = workflow.compile()
    return app

# Expose compiled app
agent_workflow = compile_agent_workflow()
