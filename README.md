# Capacitor Rating and Purchase Order Organizer

## Description
This is a personal project that I created to help automate the process of replacing capacitors in CRT Monitors and TVs. 

## Features
It is comprised of three primary parts:
- A google sheets/excel document that helps streamline the process of collecting information on capacitors and identifying likely replacement candidates
- A set of scripts to
    - collect current capacitor catalog information from the major manufacturors (chemi-con, elna, kemet, nichicon, panasonic, rubycon)
    - extract the table data from the datasheets using Claude with a focus on ESR extraction
    - cleaning and standardizing the data into a consolidated csv to be used in the replacement evaluation and purchasing decisions
- A final script to 
    - read from the capacitor information collected from the device
    - find the lowest esr or the highest ripple (depending on the priority) capacitor on DigiKey that is cost-effective
    - create a purchase order that can be copied into a cart.

## Installation

```bash
# Clone the repository
git clone https://github.com/username/project-name.git

# Navigate to the project directory
cd project-name

# Install dependencies
npm install
```

## Usage

```javascript
// Example code showing how to use the project
```

## API Reference

### Endpoint 1
```
GET /api/items
```
| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `api_key` | `string` | **Required**. Your API key |

### Endpoint 2
```
POST /api/items
```

## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

`API_KEY`

`ANOTHER_API_KEY`

## Tech Stack
- Technology 1
- Technology 2
- Technology 3

## Contributing
Contributions are always welcome!

See `CONTRIBUTING.md` for ways to get started.

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Contact
- Email: your.email@example.com
- GitHub: [@yourusername](https://github.com/yourusername)
- Twitter: [@yourusername](https://twitter.com/yourusername)
