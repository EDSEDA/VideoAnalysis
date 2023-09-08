import * as React from 'react';
import {FormControl, InputLabel, MenuItem, Select} from "@mui/material";
import {ReactComponent as Net} from '@eda/assets/net.svg';
import {ReactComponent as Clients} from '@eda/assets/clients.svg';
import {BarPlot} from "@eda/ui/plots/BarPlot.tsx";
import {LinePlot} from "@eda/ui/plots/LinePlot.tsx";
import {DatePicker} from "@mui/x-date-pickers/DatePicker";
import dayjs from "dayjs";
import {ReactComponent as HappySmile} from '@eda/assets/happySmile.svg';
import {ReactComponent as NeutralSmile} from '@eda/assets/neutralSmile.svg';
import {ReactComponent as SadSmile} from '@eda/assets/sadSmile.svg';
import classNames from 'classnames';

import styles from "./Statistics.module.css";

function randomTrandIntArr(n: number, start: number, step: number, trend: number) {
    const res = Array.from(
        {length: n},
        () => Math.max(Math.round((Math.random() - 0.5 + trend) * step), 0)
    );
    res[0] = start;
    for (let i = 1; i < res.length; i++) {
        res[i] += res[i - 1];
    }
    return res;
}

function sum(arr: number[]) {
    return arr.reduce((s, a) => s + a, 0);
}

interface Company {
    name: string,
    franchiseCount: number,
    citiesCount: number,
    clientsPerMonth: number,
}

interface Franchise {
    name: string,
    totalClients: number,
    happyClientsYear: number[],
    neutralClientsYear: number[],
    sadClientsYear: number[],
}

const companies: Company[] = [
    {
        name: 'Company 1',
        franchiseCount: 10,
        citiesCount: 5,
        clientsPerMonth: 30000,
    },
    {
        name: 'Company 2',
        franchiseCount: 3,
        citiesCount: 2,
        clientsPerMonth: 10000,
    },
];

const createFranchise = (company: Company) => {
    const total = company.clientsPerMonth * (Math.random() + 0.5);
    const neutral = randomTrandIntArr(12, Math.floor(total * Math.random()), 10000, 0);
    const happy = randomTrandIntArr(12, Math.floor((total - neutral[0]) * Math.random()), 10000, -0.1);
    const sad = randomTrandIntArr(12, Math.floor((total - neutral[0] - happy[0]) * Math.random()), 10000, 0.1);

    neutral.reverse();
    happy.reverse();
    sad.reverse();

    return {
        name: 'Franchise ' + Math.floor(Math.random() * 1000),
        totalClients: total,
        happyClientsYear: happy,
        neutralClientsYear: neutral,
        sadClientsYear: sad,
    } as Franchise;
};

export const Statistics = () => {
    const [company, setCompany] = React.useState<Company>(companies[0]);

    const franchises = React.useMemo(
        () => Array.from({length: company.franchiseCount}, () => createFranchise(company)),
        [company.name]
    );

    const [chosenFranchise, setChosenFranchise] = React.useState<Franchise>(franchises[0]);

    const franchise = React.useMemo(
        () => (franchises.includes(chosenFranchise)) ? chosenFranchise : franchises[0],
        [company, chosenFranchise]
    );

    const barPlotMock = React.useMemo(() => ({
        labels: franchises.map(franchise => franchise.name),
        datasets: [{
            data: franchises.map(franchise => franchise.totalClients),
            barThickness: 100,
        }]
    }), [franchises]);

    const linePlotMock = React.useMemo(() => ({
        labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        datasets: [
            {
                data: franchise.happyClientsYear,
                borderColor: 'rgb(69, 179, 108)',
            },
            {
                data: franchise.neutralClientsYear,
                borderColor: 'rgb(255, 209, 103)',
            },
            {
                data: franchise.sadClientsYear,
                borderColor: 'rgb(239, 70, 111)',
            },
        ],
    }), [franchise]);

    return (
        <>
            <FormControl className={styles.select}>
                <InputLabel id="company-select-label">Company</InputLabel>
                <Select
                    labelId="company-select-label"
                    id="company-select"
                    value={company.name}
                    label="Company"
                    onChange={e => {
                        const newCompany = companies.find(company => company.name === e.target.value);
                        if (newCompany) {
                            setCompany(newCompany);
                        }
                    }}
                >
                    {companies.map(company => (
                        <MenuItem key={company.name} value={company.name}>{company.name}</MenuItem>
                    ))}
                </Select>
            </FormControl>
            <div className={styles['company-info']}>
                <section className={classNames(styles.block, styles.__purple)}>
                    <div className={styles.block_img}>
                        <Net/>
                    </div>
                    <div className={styles.block_info}>
                        <h3 className={styles.block_header}>Franchise</h3>
                        <p className={styles.block_text}>{company.franchiseCount} franchise</p>
                        <p className={styles.block_text}>{company.citiesCount} cities</p>
                    </div>
                </section>
                <section className={classNames(styles.block, styles.__blue)}>
                    <div className={styles.block_img}>
                        <Clients/>
                    </div>
                    <div className={styles.block_info}>
                        <h3 className={styles.block_header}>Clients</h3>
                        <p className={styles.block_text}>average clients per year: {company.clientsPerMonth}</p>
                    </div>
                </section>
            </div>
            <div className={styles.separator}/>
            <h3>Client statistics</h3>
            <div className={styles['bar-plot']}>
                <BarPlot data={barPlotMock}/>
            </div>
            <div className={styles.separator}/>
            <div className={styles['line-plot_header-cnt']}>
                <h3 className={styles['line-plot_header']}>Emotion statistics</h3>
                <FormControl className={styles.select}>
                    <InputLabel id="franchise-select-label">Franchise</InputLabel>
                    <Select
                        labelId="franchise-select-label"
                        id="franchise-select"
                        value={franchise.name}
                        label="Franchise"
                        onChange={e => {
                            const newFranchise = franchises.find(franchise => franchise.name === e.target.value);
                            if (newFranchise) {
                                setChosenFranchise(newFranchise);
                            }
                        }}
                    >
                        {franchises.map(franchise => (
                            <MenuItem key={franchise.name} value={franchise.name}>{franchise.name}</MenuItem>
                        ))}
                    </Select>
                </FormControl>
                <DatePicker label='From' defaultValue={dayjs().subtract(7, 'days')}/>
                <DatePicker label='To' defaultValue={dayjs()}/>
            </div>
            <div className={styles['line-plot']}>
                <LinePlot data={linePlotMock}/>
            </div>
            <div className={styles['additional-info']}>
                {[
                    {title: 'Satisfied clients', text: `Total: ${sum(franchise.happyClientsYear)}`, icon: HappySmile},
                    {title: 'Neutral clients', text: `Total: ${sum(franchise.neutralClientsYear)}`, icon: NeutralSmile},
                    {title: 'Dissatisfied clients', text: `Total: ${sum(franchise.sadClientsYear)}`, icon: SadSmile},
                ].map(({title, text, icon}) => (
                    <div className={styles['additional-info_item']}>
                        {React.createElement(icon)}
                        <div>
                            <h4>{title}</h4>
                            <p>{text}</p>
                        </div>
                    </div>
                ))}
            </div>
        </>
    );
};
