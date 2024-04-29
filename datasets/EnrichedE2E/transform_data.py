import xml.etree.ElementTree as ET

# File containing XML data
file_name = 'train/8attributes.xml'

# Function to parse XML from a file and format output
def parse_and_format_xml(file_name):
    # Open the XML file and parse it
    try:
        tree = ET.parse(file_name)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        return []
    except FileNotFoundError:
        print(f"File {file_name} not found.")
        return []

    results = []
    for entry in root.findall('.//target'):
        # Extract data from XML
        inputs = {input_.get('attribute'): input_.get('value') for input_ in entry.findall('.//input')}
        
        # Build the description string, filtering out unspecified attributes
        description_parts = [
            f"name : {inputs['name']}" if 'name' in inputs else "",
            f"Type : {inputs['eatType']}" if 'eatType' in inputs else "",
            f"food : {inputs['food']}" if 'food' in inputs else "",
            f"price : {inputs['priceRange']}" if 'priceRange' in inputs else "",
            f"customer rating : {inputs['customer rating']}" if 'customer rating' in inputs else "",
            f"area : {inputs['area']}" if 'area' in inputs else "",
            f"family friendly : yes" if 'familyFriendly' in inputs and inputs['familyFriendly'] == "yes" else "",
            f"near : {inputs['near']}" if 'near' in inputs else ""
        ]

        # Filter out empty strings
        description_fields = " | ".join(part for part in description_parts if part)
        text_node = entry.find('text')
        text_value = text_node.text if text_node is not None else "Description not available"

        # Combine fields with the descriptive text
        description = f"{description_fields}||{text_value}"
        results.append(description)
    
    return results

# Extract and format entries from the XML file
formatted_entries = parse_and_format_xml(file_name)

# Write results to a text file
with open('data8.txt', 'w') as file:
    for entry in formatted_entries:
        file.write(entry + '\n')

if formatted_entries:
    print("Entries have been formatted and saved to 'formatted_entries.txt'")
else:
    print("No entries to format or save.")
