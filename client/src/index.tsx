import React from 'react'
import ReactDOM from 'react-dom/client'
import {createBrowserRouter, RouterProvider} from "react-router-dom";
import {ErrorPage} from "./app/ErrorPage.tsx";
import {Root} from "./app/Root.tsx";
import {Dashboard} from "@eda/app/Dashboard.tsx";
import {Statistics} from "@eda/app/statistics/Statistics.tsx";
import {AdapterDayjs} from '@mui/x-date-pickers/AdapterDayjs';
import {LocalizationProvider} from "@mui/x-date-pickers/LocalizationProvider";

import './index.css'
import 'dayjs/locale/de';

export const router = createBrowserRouter([
    {
        element: <Root/>,
        errorElement: <ErrorPage/>,
        children: [
            {
                path: "/",
                element: <Dashboard/>,
                errorElement: <ErrorPage/>,
            },
            {
                path: "/statistics",
                element: <Statistics/>,
                errorElement: <ErrorPage/>,
            },
        ],
    },
]);

ReactDOM.createRoot(document.getElementById('root')!).render(
    <React.StrictMode>
        <LocalizationProvider dateAdapter={AdapterDayjs} adapterLocale="de">
            <RouterProvider router={router}/>
        </LocalizationProvider>
    </React.StrictMode>,
);
