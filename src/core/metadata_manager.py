import os
import xml.etree.ElementTree as ET
import logging

logger = logging.getLogger(__name__)

# Namespaces for XMP
NS = {
    'x': 'adobe:ns:meta/',
    'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
    'xmp': 'http://ns.adobe.com/xap/1.0/'
}
# Register namespaces
for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)

class MetadataManager:
    """
    Manages XMP sidecar files for storing ratings and flags.
    """
    
    @staticmethod
    def get_xmp_path(image_path: str) -> str:
        """Get the path to the .xmp sidecar file."""
        return os.path.splitext(image_path)[0] + ".xmp"

    @staticmethod
    def read_metadata(image_path: str) -> dict:
        """
        Read rating and label from XMP sidecar.
        Returns dict with 'rating' (int) and 'label' (str).
        """
        xmp_path = MetadataManager.get_xmp_path(image_path)
        metadata = {'rating': 0, 'label': ''}
        
        if not os.path.exists(xmp_path):
            return metadata
            
        try:
            tree = ET.parse(xmp_path)
            root = tree.getroot()
            
            # Find Description element
            # Note: Parsing XMP with ElementTree can be tricky due to namespaces variations
            # This is a simplified parser
            
            # Look for Rating
            for desc in root.findall(".//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description"):
                 # Check attributes first
                rating_attr = desc.get('{http://ns.adobe.com/xap/1.0/}Rating')
                if rating_attr:
                    metadata['rating'] = int(rating_attr)
                
                label_attr = desc.get('{http://ns.adobe.com/xap/1.0/}Label')
                if label_attr:
                    metadata['label'] = label_attr
                    
        except Exception as e:
            logger.error(f"Error reading XMP {xmp_path}: {e}")
            
        return metadata

    @staticmethod
    def write_metadata(image_path: str, rating: int = None, label: str = None):
        """
        Write or update rating and label in XMP sidecar.
        """
        xmp_path = MetadataManager.get_xmp_path(image_path)
        
        # Create minimal XMP skeleton if it doesn't exist
        if not os.path.exists(xmp_path):
            xmp_str = f'''<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:xmp="http://ns.adobe.com/xap/1.0/"
    xmp:Rating="0"
    xmp:Label=""/>
 </rdf:RDF>
</x:xmpmeta>'''
            with open(xmp_path, 'w') as f:
                f.write(xmp_str)
        
        try:
            tree = ET.parse(xmp_path)
            root = tree.getroot()
            
            # Find or create Description
            desc = root.find(".//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description")
            if desc is None:
                # This should handle cases where structure is different, but for now assuming simple structure
                rdf = root.find(".//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}RDF")
                desc = ET.SubElement(rdf, "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description")
                desc.set('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about', "")
                
            if rating is not None:
                desc.set('{http://ns.adobe.com/xap/1.0/}Rating', str(rating))
            
            if label is not None:
                # Map internal flags to standard XMP labels if needed, or just store string
                desc.set('{http://ns.adobe.com/xap/1.0/}Label', label)
                
            tree.write(xmp_path, encoding="utf-8", xml_declaration=True)
            
        except Exception as e:
            logger.error(f"Error writing XMP {xmp_path}: {e}")
