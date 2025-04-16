"""
OpenAI Batch Processor for cost-efficient processing of large text batches.

This module implements OpenAI's Batch API for processing large numbers of texts 
with significant cost savings compared to synchronous API calls.
"""

import os
import json
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
import uuid

from src.utils.file_utils import ensure_dir, save_json, get_processed_item_ids
from src.utils.logging_utils import get_logger
from src.llm.model_handler import LLMHandler

# Import OpenAI library
from openai import OpenAI

# Initialize logger
logger = get_logger(__name__)

class OpenAIBatchProcessor:
    """
    Process large batches of text using OpenAI's Batch API for cost efficiency.
    Using Batch API provides approximately 50% cost savings compared to sync API calls.
    """
    
    def __init__(self):
        """Initialize the batch processor with an OpenAI client."""
        self.llm_handler = LLMHandler()
        
        # Only initialize if OpenAI is the provider
        self.openai_client = None
        if self.llm_handler.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.openai_client = OpenAI(api_key=api_key)
            else:
                logger.error("OpenAI API key not found in environment variables")
        else:
            logger.warning("Not using OpenAI provider. Batch processing will be limited.")

    def submit_batch(self, task: str, texts: List[str], 
                    item_ids: Optional[List[str]] = None, batch_size: int = 2000, 
                    batch_dir: str = "data/batch_processing",
                    parent_batch_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Submit a batch of texts for processing using OpenAI's Batch API.
        
        Args:
            task: The task name corresponding to a prompt in prompts.yaml
            texts: List of text inputs to process
            item_ids: Custom IDs to use instead of generic indices (e.g., newsIDs)
            batch_size: Maximum size for the batch
            batch_dir: Directory to store batch files and results
            parent_batch_dir: Optional existing parent batch directory to use and update
            
        Returns:
            Dict containing batch information including batch_id and status
        """
        # Validate inputs
        if not self.openai_client:
            logger.error("OpenAI client not initialized. Cannot submit batch.")
            return {"error": "OpenAI client not initialized", "status": "failed"}
        
        if not texts:
            logger.warning("Empty text list provided. Nothing to process.")
            return {"error": "Empty text list", "status": "failed"}
        
        # Check if batch size is exceeded
        if len(texts) > batch_size:
            logger.warning(f"Batch size {len(texts)} exceeds maximum configured size {batch_size}. " 
                          f"Only processing the first {batch_size} items.")
            texts = texts[:batch_size]
            if item_ids:
                item_ids = item_ids[:batch_size]
        
        # Create or use existing parent batch directory
        if parent_batch_dir and os.path.exists(parent_batch_dir):
            batch_path = parent_batch_dir
            logger.info(f"Using existing parent batch directory: {batch_path}")
            
            # Load previously processed item IDs to avoid duplicates
            # Use the get_processed_item_ids function from file_utils
            processed_ids = get_processed_item_ids(batch_path)
            
            # Filter out already processed items
            if processed_ids and item_ids:
                filtered_indices = []
                filtered_texts = []
                filtered_item_ids = []
                
                for i, item_id in enumerate(item_ids):
                    if item_id not in processed_ids:
                        filtered_indices.append(i)
                        filtered_item_ids.append(item_id)
                        filtered_texts.append(texts[i])
                
                if not filtered_texts:
                    logger.info("All items have been processed already. Nothing to do.")
                    return {"status": "skipped", "message": "All items already processed"}
                
                logger.info(f"Filtered out {len(texts) - len(filtered_texts)} already processed items.")
                texts = filtered_texts
                item_ids = filtered_item_ids
        else:
            # Create new parent batch directory
            parent_batch_id = f"parent_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            batch_path = ensure_dir(os.path.join(batch_dir, parent_batch_id))
            logger.info(f"Created new parent batch directory: {batch_path}")
            
            # Initialize parent batch metadata
            parent_metadata = {
                "parent_batch_id": parent_batch_id,
                "created_at": datetime.now().isoformat(),
                "batches": []
            }
            parent_info_path = os.path.join(batch_path, "parent_batch_info.json")
            save_json(parent_metadata, parent_info_path)
        
        # Generate a unique batch ID
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        batch_folder = ensure_dir(os.path.join(batch_path, batch_id))
        
        # Create JSONL file for batch submission
        jsonl_file = os.path.join(batch_folder, f"{batch_id}.jsonl")
        
        # Get the prompt template and model name
        prompt_template = self.llm_handler.prompts[task]
        mode_key = "full_model" if self.llm_handler.mode == "full" else "light_model"
        model_name = self.llm_handler.models["openai"][mode_key]
        
        # Prepare the batch data
        with open(jsonl_file, 'w') as f:
            for i, text in enumerate(texts):
                # Determine the item ID to use
                if item_ids and i < len(item_ids):
                    # Use the actual newsID or other custom ID
                    item_id = f"{item_ids[i]}"
                else:
                    item_id = f"item_{i}"
                
                # Format prompt with the text
                formatted_prompt = prompt_template.replace("{text}", text)
                
                # Create entry with unique ID for each item
                # Format according to OpenAI Batch API requirements
                entry = {
                    "custom_id": item_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model_name,
                        "messages": [
                            {"role": "user", "content": formatted_prompt}
                        ]
                    }
                }
                f.write(json.dumps(entry) + '\n')
        
        logger.info(f"Created batch file with {len(texts)} entries at {jsonl_file}")
        
        # Submit the batch to OpenAI
        try:
            # STEP 1: Upload the JSONL file first to get a file_id
            logger.info("Uploading batch file to OpenAI...")
            with open(jsonl_file, 'rb') as f:
                file_response = self.openai_client.files.create(
                    file=f,
                    purpose="batch"
                )
                
            file_id = file_response.id
            logger.info(f"File uploaded successfully with ID: {file_id}")
            
            # STEP 2: Create the batch using the file_id
            logger.info("Creating batch with the uploaded file...")
            response = self.openai_client.batches.create(
                input_file_id=file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
            
            # Store original items mapping for results reconstruction
            original_texts_mapping = {}
            processed_item_ids = []
            
            for i, text in enumerate(texts):
                # Use the same ID logic as above for consistency
                if item_ids and i < len(item_ids):
                    item_id = f"{item_ids[i]}"
                    processed_item_ids.append(item_id)
                else:
                    item_id = f"item_{i}"
                    
                original_texts_mapping[item_id] = text
            
            # Store batch information
            batch_info = {
                "batch_id": response.id,
                "created_at": datetime.now().isoformat(),
                "status": response.status,
                "n_items": len(texts),
                "expires_at": datetime.fromtimestamp(response.expires_at).isoformat() if response.expires_at else None,
                "original_texts": original_texts_mapping,
                "task": task,
                "model": model_name,
                "file_id": file_id,
                "using_custom_ids": item_ids is not None
            }
            
            # Save batch metadata
            metadata_file = os.path.join(batch_folder, "metadata.json")
            save_json(batch_info, metadata_file)
            
            # Update parent batch metadata
            self._update_parent_batch_metadata(batch_path, batch_id, len(processed_item_ids))
            
            logger.info(f"Successfully submitted batch {response.id} to OpenAI with {len(texts)} items")
            return batch_info
                
        except Exception as e:
            logger.error(f"Error submitting batch to OpenAI: {str(e)}")
            return {"error": str(e), "batch_id": batch_id, "status": "failed"}
    
    def _update_parent_batch_metadata(self, parent_batch_dir: str, batch_id: str, n_items: int):
        """
        Update the parent batch metadata with new batch information.
        
        Args:
            parent_batch_dir: The parent batch directory
            batch_id: The ID of the new batch
            n_items: Number of items in the batch
        """
        parent_info_path = os.path.join(parent_batch_dir, "parent_batch_info.json")
        if not os.path.exists(parent_info_path):
            logger.warning(f"Parent batch metadata file not found: {parent_info_path}")
            return
            
        try:
            with open(parent_info_path, 'r') as f:
                parent_info = json.load(f)
            
            # Add the new batch to the list
            parent_info["batches"].append({
                "batch_id": batch_id,
                "created_at": datetime.now().isoformat(),
                "n_items": n_items
            })
            
            # Update the last updated timestamp
            parent_info["last_updated"] = datetime.now().isoformat()
            
            # Save the updated metadata
            save_json(parent_info, parent_info_path)
            logger.info(f"Updated parent batch metadata with new batch: {batch_id}")
            
        except Exception as e:
            logger.error(f"Error updating parent batch metadata: {str(e)}")
    
    def check_batch_status(self, batch_id: str, 
                          batch_dir: str = "data/batch_processing") -> Dict[str, Any]:
        """
        Check the status of a submitted batch.
        
        Args:
            batch_id: The OpenAI batch ID to check
            batch_dir: Directory where batch metadata is stored
            
        Returns:
            Dict containing status information
        """
        if not self.openai_client:
            logger.error("OpenAI client not initialized. Cannot check batch status.")
            return {"error": "OpenAI client not initialized", "status": "failed"}
        
        # Find the batch metadata
        batch_metadata, batch_folder = self._find_batch_metadata(batch_id, batch_dir)
        
        if not batch_metadata or not batch_folder:
            logger.error(f"Batch metadata not found for batch_id: {batch_id}")
            return {"error": "Batch metadata not found", "status": "failed"}
        
        # Check batch status with OpenAI
        try:
            batch_info = self.openai_client.batches.retrieve(batch_id)
            
            # Update metadata with latest status
            batch_metadata["status"] = batch_info.status
            batch_metadata["last_checked"] = datetime.now().isoformat()
            
            # Save updated metadata
            metadata_file = os.path.join(batch_folder, "metadata.json")
            save_json(batch_metadata, metadata_file)
            
            logger.info(f"Batch {batch_id} status: {batch_info.status}")
            
            return {
                "batch_id": batch_id,
                "status": batch_info.status,
                "completed": batch_info.status == "completed",
                "created_at": batch_metadata.get("created_at", "unknown"),
                "last_checked": batch_metadata["last_checked"],
                "output_file_id": batch_info.output_file_id,
                "error_file_id": batch_info.error_file_id
            }
            
        except Exception as e:
            logger.error(f"Error checking batch status: {str(e)}")
            return {"error": str(e), "batch_id": batch_id, "status": "failed"}
    
    def retrieve_batch_results(self, batch_id: str, 
                              batch_dir: str = "data/batch_processing",
                              output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve and process results from a completed batch.
        
        Args:
            batch_id: The OpenAI batch ID to retrieve
            batch_dir: Directory where batch metadata is stored
            output_dir: Directory to save individual results (if None, uses batch directory)
            
        Returns:
            Dict containing status and results information
        """
        if not self.openai_client:
            logger.error("OpenAI client not initialized. Cannot retrieve batch results.")
            return {"error": "OpenAI client not initialized", "status": "failed"}
        
        # First check the status
        status_info = self.check_batch_status(batch_id, batch_dir)
        
        # If there was an error checking status or batch is not completed, return early
        if "error" in status_info or not status_info.get("completed", False):
            if "error" not in status_info:
                logger.info(f"Batch {batch_id} not yet completed. Current status: {status_info.get('status', 'unknown')}")
            return status_info
        
        # Find the batch metadata
        batch_metadata, batch_folder = self._find_batch_metadata(batch_id, batch_dir)
        
        if not batch_metadata or not batch_folder:
            logger.error(f"Batch metadata not found for batch_id: {batch_id}")
            return {"error": "Batch metadata not found", "status": "failed"}
        
        # Set up output directory
        if not output_dir:
            output_dir = os.path.join(batch_folder, "results")
            
        ensure_dir(output_dir)
        
        # Get the output file ID from the status
        output_file_id = status_info.get("output_file_id")
        if not output_file_id:
            logger.error("No output file ID available in batch status.")
            return {"error": "No output file ID available", "status": "failed"}
        
        # Download the batch output file
        output_file = os.path.join(batch_folder, f"{batch_id}_output.jsonl")
        
        if not os.path.exists(output_file):
            # Download output file if it doesn't exist
            try:
                # Download the file content using the file ID
                logger.info(f"Downloading output file with ID: {output_file_id}...")
                response = self.openai_client.files.content(output_file_id)
                
                # Write the content to a file
                with open(output_file, 'wb') as f:
                    f.write(response.content)
                
                logger.info(f"Downloaded batch output file to {output_file}")
            except Exception as e:
                logger.error(f"Error downloading batch results: {str(e)}")
                return {"error": str(e), "batch_id": batch_id, "status": "failed"}
        
        # Process results and save individual files
        results = []
        original_texts = batch_metadata.get("original_texts", {})
        
        try:
            # Read the JSONL output file and process each entry
            with open(output_file, 'r') as f:
                for line in f:
                    result = json.loads(line)
                    item_id = result.get("custom_id")
                    
                    # Get the model's response content from the response
                    if result.get("response") and result["response"].get("choices"):
                        content = result["response"]["choices"][0].get("message", {}).get("content", "")
                    else:
                        logger.warning(f"No valid content found for item {item_id}")
                        content = ""
                    
                    # Skip if item_id not in original texts
                    if item_id not in original_texts:
                        continue
                        
                    # Get original index from item_id
                    index = item_id.replace("item_", "")
                    result_file = os.path.join(output_dir, f"result_{index}.json")
                    
                    # Try to parse JSON from content if available
                    try:
                        if "```json" in content and "```" in content.split("```json")[1]:
                            json_str = content.split("```json")[1].split("```")[0].strip()
                            parsed_content = json.loads(json_str)
                        else:
                            parsed_content = json.loads(content.strip())
                    except json.JSONDecodeError:
                        parsed_content = {"raw_output": content}
                    
                    # Save the result
                    save_json(parsed_content, result_file)
                    
                    # Record the result
                    results.append({
                        "item_id": item_id,
                        "result_file": result_file
                    })
        
            # Update metadata with completion info
            batch_metadata["status"] = "completed"
            batch_metadata["completed_at"] = datetime.now().isoformat()
            batch_metadata["results_path"] = output_dir
            batch_metadata["n_results"] = len(results)
            
            # Save updated metadata
            metadata_file = os.path.join(batch_folder, "metadata.json")
            save_json(batch_metadata, metadata_file)
            
            logger.info(f"Successfully processed {len(results)} results from batch {batch_id}")
            
            return {
                "batch_id": batch_id,
                "status": "completed",
                "completed": True,
                "n_results": len(results),
                "results_path": output_dir
            }
            
        except Exception as e:
            logger.error(f"Error processing batch results: {str(e)}")
            return {"error": str(e), "batch_id": batch_id, "status": "failed"}
            
    def _find_batch_metadata(self, batch_id: str, batch_dir: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Find metadata file for a specific batch ID.
        
        Args:
            batch_id: Batch ID to find
            batch_dir: Directory to search
            
        Returns:
            tuple: (batch_metadata, batch_folder) or (None, None) if not found
        """
        # First, check if there's a direct path match - the batch_id is actually the name of a batch folder
        # Check inside batch_dir
        direct_path = os.path.join(batch_dir, batch_id)
        if os.path.isdir(direct_path) and os.path.exists(os.path.join(direct_path, "metadata.json")):
            metadata_path = os.path.join(direct_path, "metadata.json")
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    return metadata, direct_path
            except Exception as e:
                logger.error(f"Error reading metadata from direct path {direct_path}: {str(e)}")
        
        # Check inside parent batch directories
        for parent_dir in os.listdir(batch_dir):
            parent_path = os.path.join(batch_dir, parent_dir)
            if os.path.isdir(parent_path) and parent_dir.startswith("parent_batch_"):
                direct_path = os.path.join(parent_path, batch_id)
                if os.path.isdir(direct_path) and os.path.exists(os.path.join(direct_path, "metadata.json")):
                    metadata_path = os.path.join(direct_path, "metadata.json")
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                            return metadata, direct_path
                    except Exception as e:
                        logger.error(f"Error reading metadata from direct path {direct_path}: {str(e)}")
        
        # If no direct folder match, fall back to looking for metadata file with matching batch_id
        logger.info(f"No direct folder match for batch ID {batch_id}, searching through metadata files...")
        for dir_path, _, files in os.walk(batch_dir):
            if "metadata.json" in files:
                metadata_path = os.path.join(dir_path, "metadata.json")
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        if metadata.get("batch_id") == batch_id:
                            return metadata, dir_path
                except Exception:
                    continue
                    
        return None, None

# Example Usage
if __name__ == "__main__":
    processor = OpenAIBatchProcessor()
    
    # Example 1: Submit a batch
    texts = [
        """'HALIFAX, NS, Feb. 21, 2024 /CNW/ - Fortune Bay Corp. (TSXV: FOR) (FWB: 5QN) (OTCQB: FTBYF) ("Fortune Bay" or the "Company") is pleased to announce the acquisition of two additional uranium projects through staking on the north-central margin of the Athabasca Basin, in proximity to the Company\'s recently announced Spruce, Pine and Aspen Uranium Projects (Figure 1). Gareth Garlick, Technical Director for Fortune Bay, commented "The acquisition of the Birch and Fir projects adds to our growing uranium portfolio of newly acquired, 100% owned projects on the north-central margin of the Athabasca Basin. This extensive portfolio now totals five new uranium projects covering over 40,000 hectares and provides Fortune Bay with further opportunity to create value through exploration and/or transactional success. The Birch and Fir projects have known uranium endowment with historical occurrences of up to 55.1% U(3) O(8) , in addition to Rare Earth Element potential with historical outcrop grades of up to 2.4% Total Rare Earth Elements." In addition, the Company reports on progress for its Murmac and Strike Uranium Projects ("Murmac" and "Strike"), located on the north-western margin of the Athabasca Basin of Saskatchewan, for which an Option Agreement was recently signed. Dale Verran, CEO for Fortune Bay, commented, "We are delighted with the progress Aero Energy Limited has made for Murmac and Strike, with exploration drilling planned to commence in the coming months. Together with Aero\'s award-winning technical advisory team, operational planning and prioritization of drill targets is well underway. Murmac and Strike present significant opportunity for the discovery of high-grade, basement-hosted uranium and we look forward to working with Aero to advance exploration, while retaining upside in future discovery." Newly Staked Uranium Projects Birch Project The Birch Uranium Project ("Birch") comprises four mineral claims totalling 5,751 hectares located approximately 35 kilometres north of the Athabasca Basin margin, and has potential for unconformity-related, basement-hosted deposits and bulk tonnage Rossing-style uranium deposits. Birch remains underexplored with no drilling to date. As follow-up to Government regional airborne radiometric surveys, historical prospecting between 1968 and 1970 identified widespread and voluminous uranium-bearing pegmatites in the Box Lake Area (Figure 1 -- Birch Project Block A). Individual pegmatites were traced over lengths exceeding 150 metres and widths exceeding 30 metres, with average sampled grades ranging from 200 to 300 ppm. Notably, higher grades were also recorded where structures could be sampled with grades between 0.22% and 0.36% U(3) O(8) . Uranium mineralization included uraninite and carnotite staining. Exploration is warranted to, 1) re-investigate the pegmatite uranium occurrences, 2) establish the nature and extent of the structurally-associated higher grade uranium mineralization (which would be expected to predominantly occur in low-lying areas with sediment/water cover), and 3) explore for extensions of these two types of mineralization to the south-southwest into a large favorable hinge zone target area of structural complexity which appears to be largely unexplored. To the south (Figure 1 -- Birch Project Block B), ground radiometric anomalies were identified during historical prospecting at Miller Lake (equivalent uranium grades of up to 861 ppm eU), within a smaller hinge zone, also warranting modern follow-up. Fir Project The Fir Uranium Project ("Fir") comprises a single mineral claim of 794 hectares located approximately ten kilometres north of the Athabasca Basin margin, and has potential for unconformity-related, basement-hosted deposits and bulk tonnage Rossing-style uranium deposits. Fir remains underexplored with no drilling to date. A historical pebble sample assayed 55.06% U(3) O(8) , one of several radioactive pebbles found in a low-lying area covered by muskeg. An additional historical uranium occurrence of 0.24% U(3) O(8) from an outcrop sample is present 400 metres to the southeast. The area is characterized by interpreted structural complexity at the intersection of east and northeast trending structures. Follow-up of the occurrence was limited to localized prospecting, and the occurrence warrants more detailed systematic follow-up, particularly in low-lying areas with surficial cover where structures are likely to exist. In addition to the uranium potential, pegmatite outcrops enriched in Rare Earth Elements ("REE") were discovered during historical prospecting for uranium. This included an outcrop sample of 2.4% Total Rare Earth Elements ("TREE"), and a 400 x 200 m outcrop of "white granodiorite" displaying broadly elevated TREE content and highlight grades from four samples of 1.1% to 1.9% TREE. High-value Nd and Pr account for approximately 20% of the TREE content. These historical REE occurrences warrant follow-up to determine the extent and grades of the mineralization. Far northern Saskatchewan has a precedent for high-grade pegmatite REE deposits, including Alces Lake (grades up to 30% TREE), Bear Lake (16% TREE rock sample), and Hoidas Lake (historical NI 43-101 mineral resource estimate including 2.6 million tonnes at 2% TREE). Similar to other recently announced uranium projects, Fortune Bay may seek to find a suitable partner/s to advance the Birch and Fir Projects through an earn-in, or similar agreement, that provides Fortune Bay with upside in future discovery. Murmac and Strike Update: -- On December 18, 2023 Fortune Bay Corp. announced that it had entered into \n      an "Option Agreement" whereby 1443904 B.C. Ltd., an arms-length private \n      company, was granted the right to acquire up to a 70% interest in the \n      wholly-owned Murmac and Strike Uranium Projects over a \n      three-and-a-half-year period by funding C$6 million in exploration \n      expenditures, making cash payments totalling C$1.35 million, and issuing \n      C$2.15 million in common shares following completion of a going public \n      transaction (see News Release for further details) \n \n   -- On February 8, 2024, Aero Energy Limited (TSXV: AERO) (OTC Pink: AAUGF) \n      (FSE: 13L0) ("Aero") completed the acquisition of 1443904 B.C. Ltd. \n      pursuant to the terms of a share purchase agreement, thereby completing \n      the going public transaction. Concurrent with completion of the \n      acquisition, the Company has changed its name from "Angold Resources \n      Ltd." to "Aero Energy Limited". \n \n   -- On February 13, 2024 Aero announced a non-brokered private placement for \n      aggregate gross proceeds of $5,000,000. The gross proceeds received from \n      the sale of the Flow-Through units and the Charity Units will be used for \n      work programs on Aero\'s optioned properties including Murmac, Strike and \n      Sun Dog (owned by Standard Uranium Ltd.), all located along the \n      northwestern margin of the Athabasca Basin of Saskatchewan. \n \n   -- In accordance with the Option Agreement Fortune Bay has received the \n      initial $200,000 cash payment and the initial $200,000 in common shares \n      (1,333,333 shares using a transaction price of $0.15 per share). \n \n   -- Planning is ongoing for drilling programs to commence at Murmac and \n      Strike during 2024 on numerous high-priority targets that have been \n      identified with the potential for a high-grade basement-hosted uranium \n      discovery. Correction to Previously Announced Deferred Share Unit Grant On January 10, 2024, the Company announced the grant of Deferred Share Units ("DSUs") to certain directors and officers.  It was noted that 150,000 DSUs were granted to the directors of the Company to settle director fees for the year ended December 31, 2023.  The number of DSUs granted should have been stated to be 200,000.  The DSUs will vest in accordance with the Company\'s deferred share unit plan. Technical Disclosure The historical results contained within this news release have not been verified and there is a risk that any future confirmation work and exploration may produce results that substantially differ from the historical results. The Company also cautions that historical results on adjacent properties are not necessarily indicative of the results that may be achieved on the Project. The Company considers these historical results relevant to assess the mineralization and economic potential of the property. Further details regarding the historical occurrences mentioned in this news release can be found within the Saskatchewan Mineral Deposit Index ("SMDI") using the reference numbers provided in Figure 1. Additional information has been obtained from historical reports found within Saskatchewan Mineral Assessment Database ("SMAD") with the following Assessment File Numbers 74O10-0002, 74O10-0003, 74O10-0008 (Birch Project), and 74O09-0023, MAW02300 (Fir Project). Details regarding the other REE occurrences in far northern Saskatchewan can be found using the following references/links: Alces Lake: Appia Rare Earths and Uranium Corp. (https://appiareu.com/alces-lake/); Bear Lake: SMDI#3571; Hoidas Lake: SMDI#1612. Qualified Person The technical and scientific information in this news release has been reviewed and approved by Gareth Garlick, P.Geo., Technical Director of the Company, who is a Qualified Person as defined by NI 43-101. Mr. Garlick is an employee of Fortune Bay and is not independent of the Company under NI 43-101. About Fortune Bay Corp. (MORE TO FOLLOW) Dow Jones Newswires February 21, 2024 07:00 ET (12:00 GMT)'""",
        """'VANCOUVER, BC, March 8, 2024 /CNW/ - Applied Graphite Technologies Corporation (formerly Audrey Capital Corporation) (TSXV: AGT) (the "Corporation"), is pleased to announce that it has closed its previously announced Qualifying Transaction (as defined in Policy 2.4 Capital Pool Companies of the TSX Venture Exchange (the "TSXV")). The Qualifying Transaction proceeded by way of a three-cornered amalgamation pursuant to which Applied Graphite Technologies Corporation, a private company incorporated under the Business Corporations Act (British Columbia) ("AGT"), amalgamated with 1445056 B.C. Ltd., to become a wholly-owned subsidiary of the Corporation ("AGT Resources"). The name of the amalgamated subsidiary corporation is "AGT Resources Corporation". The Corporation, as the resulting issuer (the "Resulting Issuer"),  will continue the business of AGT and has changed its name to Applied Graphite Technologies Corporation and its ticker symbol to "AGT" on the TSXV. The Corporation has consolidated its 20,000,000 outstanding common shares on a 1.5:1 ratio, which results in 13,333,333 common shares outstanding post-consolidation. The Corporation issued approximately 8,200,605 common shares to the shareholders of AGT, and a total of 21,533,938 common shares of the Resulting Issuer (the "Resulting Issuer Shares") are issued and outstanding, along with 1,333,333 stock options, 333,333 broker warrants, and 1,366,454 warrants all exercisable at $0.15 per common share. 3,950,723 of the Resulting Issuer Shares will be held for up to 36 months from the date of issuance of the Final Exchange Bulletin by the TSXV (the "Final Bulletin"), pursuant to a TSXV Form 5D Escrow Agreement (the "Escrow Agreement"). The Corporation\'s new CUSIP number will be 03820A109 and its new ISIN number will be CA03820A1093. Former registered holders of pre-consolidation common shares of the Corporation will be receiving by mail, from Olympia Trust, the Corporation\'s transfer agent, a letter of transmittal with instructions on how to remit former common shares of the Corporation for post-consolidation common shares of the Corporation. For further information, please refer to the Corporation\'s profile on SEDAR+ at www.sedarplus.ca, the Filing Statement dated February 29, 2024 regarding the Qualifying Transaction, as well as the press releases dated June 26, 2023 and February 29, 2024. Trading of the common shares of AGT will remain halted in connection with the dissemination of this press release and will recommence at such time as the Exchange may determine, having regard to the completion of certain requirements pursuant to Exchange Policy 2.4. All directors and officers of the Corporation have resigned effective March 7, 2024. The directors of the Resulting Issuer are Don Baxter, Ian Harris, Rodney Stevens, James Ruane and Chaanaka Abeyratne. These directors shall hold office until the first annual general meeting of the shareholders of the Resulting Issuer following closing, or until their successors are duly appointed or elected. The officers of the Resulting Issuer are Don Baxter as Chief Executive Officer, James Ruane as Chairman of the Board, Sunil Sharma as Chief Financial Officer and Melissa Martensen as Corporate Secretary. Final acceptance of the Qualifying Transaction will occur upon the issuance of the Final Bulletin. Subject to final acceptance by the TSXV, the Corporation will be classified as a Tier 2 mining issuer pursuant to TSXV policies. The Common Shares are expected to commence trading on the TSXV under the symbol "AGT" at the opening of the markets on March 12, 2024. Pursuant to the Qualifying Transaction, Donald Baxter acquired control over 2,837,845 Resulting Issuer Shares (and no other securities of the Resulting Issuer) all of which were issued in exchange for the common shares of AGT controlled by Mr. Baxter prior to completion of the Qualifying Transaction. Mr. Baxter exercises control over 2,837,845 (13.18%) of the issued and outstanding Resulting Issuer Shares. Mr. Baxter currently does not have any plan to acquire or dispose of additional securities of the Corporation. However, Mr. Baxter may acquire additional securities of the Corporation, dispose of some or all of the existing or additional securities he holds or will hold, or may continue to hold his current position, depending on market conditions, reformulation of plans and/or other relevant factors. All of the Resulting Issuer Shares controlled by Mr. Baxter are subject to the terms of the Escrow Agreement. Immediately prior to completion of the Qualifying Transaction, Ian Slater exercised control over 4,000,000 pre-consolidation common shares of the Corporation ("CPC Shares") and 600,000 pre-consolidation options of the Corporation ("CPC Options"), representing approximately 20% of the issued and outstanding CPC Shares on a non-diluted basis and approximately 22.33% of the issued and outstanding CPC Shares on a on a partially diluted basis. Upon completion of the Qualifying Transaction, Mr. Slater exercises control over 2,666,667 Resulting Issuer Shares, 400,000 options of the Resulting Issuer and 1,366,454 Resulting Issuer Share purchase warrants, representing approximately 12.38% of the issued and outstanding Resulting Issuer Shares on a non-diluted basis and approximately 20.21% of the Resulting Issuer Shares on a partially diluted basis. Mr. Slater currently does not have any plan to acquire or dispose of additional securities of the Corporation. However, Mr. Slater may acquire additional securities of the Corporation, dispose of some or all of the existing or additional securities he holds or will hold, or may continue to hold his current position, depending on market conditions, reformulation of plans and/or other relevant factors. All of the Resulting Issuer Shares and options of the Resulting Issuer controlled by Mr. Slater are subject to the terms of a previously executed TSXV Form 2F CPC Escrow Agreement (the "CPC Escrow Agreement"). Immediately prior to completion of the Qualifying Transaction, Paul Beattie exercised control over 4,000,000 pre-consolidation CPC Shares and 600,000 CPC Options, representing approximately 20% of the issued and outstanding CPC Shares on a non-diluted basis and approximately 22.33% of the issued and outstanding CPC Shares on a on a partially diluted basis. Upon completion of the Qualifying Transaction, Mr. Beattie exercises control over 2,666,667 Resulting Issuer Shares and 400,000 options of the Resulting Issuer, representing approximately 12.38% of the issued and outstanding Resulting Issuer Shares on a non-diluted basis and approximately 13.98% of the Resulting Issuer Shares on a partially diluted basis. Mr. Beattie currently does not have any plan to acquire or dispose of additional securities of the Corporation. However, Mr. Beattie may acquire additional securities of the Corporation, dispose of some or all of the existing or additional securities he holds or will hold, or may continue to hold his current position, depending on market conditions, reformulation of plans and/or other relevant factors. All of the Resulting Issuer Shares and options of the Resulting Issuer controlled by Mr. Beattie are subject to the terms of the CPC Escrow Agreement. Immediately prior to completion of the Qualifying Transaction, Jackie Cheung exercised control over 4,000,000 CPC Shares (and no other securities of the Corporation), representing approximately 20% of the issued and outstanding CPC Shares. Upon completion of the Qualifying Transaction, Mr. Cheung exercises control over 2,666,667 Resulting Issuer Shares, representing approximately 12.38% of the issued and outstanding Resulting Issuer Shares. Mr. Cheung currently does not have any plan to acquire or dispose of additional securities of the Corporation. However, Mr. Cheung may acquire additional securities of the Corporation, dispose of some or all of the existing or additional securities he holds or will hold, or may continue to hold his current position, depending on market conditions, reformulation of plans and/or other relevant factors. All of the Resulting Issuer Shares controlled by Mr. Cheung are subject to the terms of the CPC Escrow Agreement. The foregoing disclosure is being disseminated pursuant to National Instrument 62-103 -- The Early Warning System and Related Take-Over Bid and Insider Reporting. Copies of the early warning reports with respect to the foregoing will appear on the Corporation \'s SEDAR+ profile at www.sedarplus.ca and may also be obtained by contacting the Corporation at 604-638-2545. Applied Graphite Technologies Corporation owns a 90% ownership interest in C-Tech Ceylon (Private) Limited, a corporation incorporated pursuant to the laws of Sri Lanka, which in turn owns a 100% ownership interest in the Dodangaslanda Graphite Properties in Sri Lanka. The Dodangaslanda Properties are on private land in the heart of the vein graphite district, with historical workings and vein graphite outcrops. Vein graphite is naturally high grade (+95% carbon content in the ground) and does not require primary processing. Testing of vein graphite in lithium-ion battery anodes has shown very high capacities, performing better than synthetic graphite. Natural vein graphite has a far superior ESG footprint than synthetic and is cheaper without compromising performance. The technical information in this news release has been prepared by Don Baxter, P.Eng., a "qualified person" as defined in National Instrument 43-101 Standards of Disclosure for Mineral Projects ("NI 43-101"). Cautionary Note Investors are cautioned that, except as disclosed in the continuous disclosure document containing full, true and plain disclosure regarding the Transaction, required to be filed with the securities regulatory authorities having jurisdiction over the affairs of the Corporation, any information released or received with respect to the Transaction may not be accurate or complete and should not be relied upon. The trading in the securities of the Corporation on the TSXV should be considered highly speculative. (MORE TO FOLLOW) Dow Jones Newswires March 08, 2024 09:00 ET (14:00 GMT)'""",
        """'Deloitte Acquires Gryphon Scientific Business to Expand Security, Science and Public Health Capabilities PR Newswire ARLINGTON, Va., April 29, 2024 Acquisition will strengthen Deloitte team with a complement of data-driven biosecurity detection, prevention and emergency response solutions ARLINGTON, Va., April 29, 2024  /PRNewswire/ -- Deloitte announced that it has acquired substantially all of the assets of Gryphon Scientific, LLC (Gryphon), a leader in biosafety, biosecurity, and all-hazards preparedness and response, with experience in using artificial intelligence (AI) to enhance security and safety. Deloitte\'s competitive edge will be strengthened by Gryphon\'s multidisciplinary team who will be joining Deloitte, comprising scientists, programmers, and policy and planning professionals with experience in data science, scientific communications, modeling and risk assessment. Gryphon\'s cadre of specialists will enhance Deloitte\'s capability to address its clients\' most complex mission challenges, from informing safe policy and practice around novel technologies and AI to strengthening capacities to prevent, detect and respond to infectious disease threats across the globe. "The addition of Gryphon\'s leadership and key professionals to the Deloitte team represents a significant enhancement to Deloitte\'s data analytics and advanced technology capabilities. Our federal health practice is excited to lead the way for U.S. government and public services (GPS) to push the boundaries and bring our clients to the forefront of AI-enabled, mission-driven work," said Beth Meagher, principal, Deloitte Consulting LLP and U.S. federal health sector leader. "This enhances the types of data-driven technology and scientific experience that we can offer to federal agencies and strengthens our ability to support government leaders in their efforts to safeguard the security of our nation and the health and safety of our people." Over the past two decades, Gryphon has supported senior decision makers within government and the commercial sector in evaluating emerging technologies and understanding how to enable the rapid development of these critical tools, while also safeguarding against their associated risks. Gryphon is at the forefront of AI safety, especially understanding how AI can change the risk landscape of biological and chemical threats. Together, Gryphon and Deloitte leaders will continue to develop practical AI applications in health, encourage multi-sector collaboration, and identify and manage risks to create trustworthy AI solutions through the Federal Health AI Accelerator at Deloitte. "As leaders in the GPS space for almost 20 years, we\'ve focused on applying our scientific understanding toward preventing and solving complex problems affecting public health and safety," said Dr. Rocco Casagrande, Gryphon\'s founder and executive chairman. "Together with Deloitte, we\'ll further our mission of addressing some of the most critical issues in the life sciences facing the world today." At Deloitte, Gryphon\'s technical specialists will advance Deloitte\'s life science and public health preparedness and response capabilities, catalyzing the growth of Deloitte\'s biosecurity, lab safety, data science, and global and domestic emergency preparedness and response capabilities. Gryphon\'s experienced scientists and planning professionals will also help Deloitte clients prepare for biological emergencies and biothreats -- and strengthen public health and safety nationwide. "We\'re proud to welcome Gryphon\'s talented scientists, data analysts, and public health specialists to Deloitte," said Jason Salzetti, principal, Deloitte Consulting LLP and GPS industry leader. "Their extensive experience will further bolster our leadership in artificial intelligence and help our clients address biosafety threats." As used herein, "Deloitte" refers to Deloitte Consulting LLP, a subsidiary of Deloitte LLP. Please see www.deloitte.com/us/about for a detailed description of our legal structure. Certain services may not be available to attest clients under the rules and regulations of public accounting. View original content to download multimedia:https://www.prnewswire.com/news-releases/deloitte-acquires-gryphon-scientific-business-to-expand-security-science-and-public-health-capabilities-302129084.html SOURCE Deloitte /CONTACT: Karen Walsh, Public Relations, Deloitte Consulting LLP, +1 717-982-8194, karwalsh@deloitte.com  (END) Dow Jones Newswires April 29, 2024 09:00 ET (13:00 GMT)'"""
    ]
    
    # For small number of texts, use submit_batch
    batch_info = processor.submit_batch("v4", texts)
    print(f"Batch submitted: {batch_info}")
    
    # For large number of texts, use submit_multiple_batches
    # large_texts = ["text1", "text2", ..., "text20000"]
    # all_batches = processor.submit_multiple_batches("v4", large_texts, batch_size=2000)
    # print(f"Submitted {all_batches['num_batches']} batches with parent ID: {all_batches['parent_batch_id']}") 