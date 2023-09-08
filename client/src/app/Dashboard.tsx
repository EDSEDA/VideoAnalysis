import {StackedBarPlot} from "@eda/ui/plots/StackedBarPlot.tsx";
import {SimpleLinePlot} from "@eda/ui/plots/SimpleLinePlot.tsx";
import Autocomplete from "@mui/material/Autocomplete";
import TextField from "@mui/material/TextField";
import {DatePicker} from '@mui/x-date-pickers/DatePicker';
import dayjs from "dayjs";
import * as React from 'react';
import {ReactComponent as HappySmile} from '@eda/assets/happySmileBubble.svg';
import {ReactComponent as NeutralSmile} from '@eda/assets/neutralSmileBubble.svg';
import {ReactComponent as SadSmile} from '@eda/assets/sadSmileBubble.svg';
import {ReactComponent as People} from '@eda/assets/people.svg';

import styles from './Dashboard.module.css';

function randomIntArr(n: number, min: number, max: number) {
    return Array.from({length: n}, () => Math.floor(Math.random() * (max - min) + min));
}

const barPlotMockData = {
    labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
    datasets: [
        {
            label: 'Franchise 1',
            data: randomIntArr(12, 50000, 100000),
            backgroundColor: 'rgb(147, 223, 246)',
            borderRadius: 5,
            barThickness: 30,
        },
        {
            label: 'Franchise 2',
            data: randomIntArr(12, 50000, 100000),
            backgroundColor: 'rgb(255, 227, 162)',
            borderRadius: 5,
            barThickness: 30,
        },
        {
            label: 'Franchise 3',
            data: randomIntArr(12, 50000, 100000),
            backgroundColor: 'rgb(255, 154, 124)',
            borderRadius: 5,
            barThickness: 30,
        },
        {
            label: 'Franchise 4',
            data: randomIntArr(12, 50000, 100000),
            backgroundColor: 'rgb(176, 198, 255)',
            borderRadius: 5,
            barThickness: 30,
        },
    ]
};

const BarPlotOptions = barPlotMockData.datasets.map(dataset => dataset.label);

function randomTrandIntArr(n: number, start: number, step: number, trend: number) {
    const res = Array.from(
        {length: n},
        () => Math.round((Math.random() - 0.5 + trend) * step)
    );
    res[0] = start;
    for (let i = 1; i < res.length; i++) {
        res[i] += res[i - 1];
    }
    return res;
}

const linePlotData = [
    {
        icon: HappySmile,
        data: {
            label: 'Satisfied customers',
            data: randomTrandIntArr(30, 1000, 10, 0.01),
            borderColor: 'rgb(69, 179, 108)',
            fill: true,
            backgroundColor: 'rgba(69, 179, 108, 0.2)',
            cubicInterpolationMode: 'monotone',
        },
    },
    {
        icon: NeutralSmile,
        data: {
            label: 'Neutral customers',
            data: randomTrandIntArr(30, 1000, 10, 0),
            borderColor: 'rgb(255, 209, 103)',
            fill: true,
            backgroundColor: 'rgba(255, 209, 103, 0.2)',
            cubicInterpolationMode: 'monotone',
        }
    },
    {
        icon: SadSmile,
        data: {
            label: 'Dissatisfied customers',
            data: randomTrandIntArr(30, 1000, 10, -0.01),
            borderColor: 'rgb(239, 70, 111)',
            fill: true,
            backgroundColor: 'rgba(239, 70, 111, 0.2)',
            cubicInterpolationMode: 'monotone',
        },
    },
] as const;

export const Dashboard = () => (
    <>
        <div className={styles.header_cnt}>
            <h3 className={styles.header_text}>Client statistics</h3>
            <Autocomplete
                options={BarPlotOptions}
                className={styles.header_picker}
                renderInput={(params) => (
                    <TextField {...params} label="All franchise"/>
                )}
            />
            <DatePicker label='From' defaultValue={dayjs().subtract(7, 'days')}/>
            <DatePicker label='To' defaultValue={dayjs()}/>
        </div>
        <section className={styles.plot}>
            <StackedBarPlot data={barPlotMockData}/>
        </section>
        <div className={styles.separator}/>
        <section className={styles.info}>
            {linePlotData.map(({data, icon}) => (
                <div key={data.label} className={styles.info_block}>
                    <h3 className={styles.block_header}>
                        {React.createElement(icon)}
                        {data.label}
                    </h3>
                    <div>
                        <SimpleLinePlot data={{
                            labels: data.data,
                            datasets: [
                                data
                            ],
                        }}/>
                    </div>
                    <p className={styles.block_footer}>
                        {data.data[data.data.length - 1]} people
                    </p>
                </div>
            ))}
            <div className={styles.info_block}>
                <h3 className={styles.block_header}>
                    <People/>
                    Employees
                </h3>
                <div>
                    <p className={styles.emphasized}>120 people</p>
                    <p>Total employees</p>
                </div>
                <div>
                    <p className={styles.emphasized}>20 people</p>
                    <p>Managers</p>
                </div>
            </div>
        </section>
    </>
);
