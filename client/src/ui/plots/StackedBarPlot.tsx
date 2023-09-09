import {registerables, Chart as ChartJS} from 'chart.js';
import {Chart} from "react-chartjs-2";
import type {ChartProps} from "react-chartjs-2";

ChartJS.register(...registerables);

export type StackedBarPlotProps = Pick<ChartProps<'bar', (number | [number, number] | null)[], unknown>, 'data'>;

export const StackedBarPlot = (props: StackedBarPlotProps) => (
    <Chart
        id='stackedBarPlot'
        type='bar'
        data={props.data}
        options={{
            maintainAspectRatio: false,
            scales: {
                x: {
                    stacked: true,
                },
                y: {
                    stacked: true
                },
            },
            plugins: {
                legend: {
                    position: 'bottom',
                    align: 'start',
                },
            },
        }}
    />
);