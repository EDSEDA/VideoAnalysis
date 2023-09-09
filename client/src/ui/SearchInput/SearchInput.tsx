import IconButton from "@material-ui/core/IconButton";
import SearchIcon from "@material-ui/icons/Search";
import TextField from "@mui/material/TextField";

export const SearchInput = () => (
    <TextField
        label="Search"
        InputProps={{
            endAdornment: (
                <IconButton>
                    <SearchIcon/>
                </IconButton>
            ),
        }}
    />
);
