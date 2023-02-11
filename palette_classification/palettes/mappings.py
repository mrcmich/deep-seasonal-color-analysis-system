REFERENCE_PALETTES_DESCRIPTIONS = ['autumn', 'spring', 'summer', 'winter']

# mappings from palette id to palette description and vice versa;
# ids are used instead of descriptions when assigning a season palette to dress code dataset instances.
ID_DESC_MAPPING = { id: description for id, description in enumerate(REFERENCE_PALETTES_DESCRIPTIONS) }
DESC_ID_MAPPING = { description: id for id, description in ID_DESC_MAPPING.items() }